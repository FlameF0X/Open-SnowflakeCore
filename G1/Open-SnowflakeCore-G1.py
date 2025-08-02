import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
from datasets import load_dataset
import random
import os
import json
import math
from typing import Tuple, Dict

# ==============================================================================
# 0. Configuration
# ==============================================================================

class Config:
    """A class to hold all training and model configuration parameters.
    This configuration targets a model with approximately 10 million parameters.
    """
    MODEL_NAME = "gpt2"
    DATASET_NAME = "common-pile/wikimedia_filtered"
    BATCH_SIZE = 1
    EPOCHS = 5
    # Adjusted hyperparameters for a smaller, ~10M parameter model
    MAX_LENGTH = 1024 # Reduced context window
    EMBED_DIM = 96 # Much smaller embedding dimension
    NUM_HEADS = 4 # Must divide EMBED_DIM evenly
    NUM_LAYERS = 8 # Fewer transformer blocks
    FFN_DIM = 384 # 4 * EMBED_DIM is standard
    
    LEARNING_RATE = 2e-4
    GRAD_ACCUM_STEPS = 32
    SAVE_PATH = "SnowflakeCore-G1-Small"
    VAL_SPLIT_RATIO = 0.05
    EARLY_STOPPING_PATIENCE = 1
    EARLY_STOPPING_MIN_DELTA = 0.001
    VALIDATE_EVERY_N_STEPS = 500
    STEP_EARLY_STOPPING_PATIENCE = 5
    STEP_EARLY_STOPPING_MIN_DELTA = 0.001
    DROPOUT = 0.1

    def get_device_and_dtype(self) -> Tuple[torch.device, torch.dtype]:
        """Determine the appropriate device and data type for training."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if device.type == 'cuda' and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Using bfloat16 (BF16) precision.")
        elif device.type == 'cuda':
            dtype = torch.float16
            print("Using float16 (FP16) precision.")
        else:
            dtype = torch.float32
            print("Using float32 (FP32) precision (BF16/FP16 not supported or on CPU).")
        
        return device, dtype

# ==============================================================================
# 1. Dataset and Data Loading
# ==============================================================================

class TextDataset(Dataset):
    """
    A custom PyTorch Dataset for handling text data from the Hugging Face
    datasets library.
    """
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a tokenized item from the dataset.
        
        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the tokenized
                                               input IDs and the attention mask.
        """
        item = self.dataset[idx]
        if "text" not in item or not isinstance(item["text"], str):
            raise ValueError(f"Sample at idx {idx} does not have a 'text' field. Got: {item}")
        
        text = item["text"]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)
        return input_ids, attention_mask

def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates a batch of tokenized text for language modeling.
    For language modeling, the labels are the same as the input IDs.
    """
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    labels = input_ids.clone()
    return input_ids, attention_mask, labels

# ==============================================================================
# 2. Model Architecture
# ==============================================================================

class FusedSelfAttention(nn.Module):
    """
    A fused self-attention module that combines QKV projections for efficiency.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == embed_dim), "embed_dim must be divisible by num_heads"
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs a forward pass of the self-attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape `[batch_size, sequence_length, embed_dim]`.
            attn_mask (torch.Tensor, optional): Causal attention mask. Defaults to None.
            key_padding_mask (torch.Tensor, optional): Mask for padded tokens. Defaults to None.
        
        Returns:
            torch.Tensor: The output tensor after attention.
        """
        B, T, C = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(0).to(attn_scores.dtype)
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = attn_probs @ v
        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)
        
        return self.out_proj(attn_output)

class GPTBlock(nn.Module):
    """A single Transformer block (LayerNorm -> Attention -> LayerNorm -> MLP)."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = FusedSelfAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for the GPTBlock."""
        h = self.ln1(x)
        attn_output = self.attn(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(attn_output)
        x = x + self.dropout2(self.mlp(self.ln2(x)))
        return x

class SnowflakeCoreG1(nn.Module):
    """
    The main SnowflakeCoreG1 model architecture, a Transformer-based decoder-only model.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, max_length: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GPTBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_length = max_length
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the SnowflakeCoreG1 model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape `[batch_size, sequence_length]`.
            attention_mask (torch.Tensor, optional): Mask for padded tokens. Defaults to None.
        
        Returns:
            torch.Tensor: Logits for the next token prediction.
        """
        B, T = input_ids.size()
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = self.dropout(x)

        # Create a causal mask for the decoder-only transformer
        causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))
        
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ==============================================================================
# 3. Trainer Class
# ==============================================================================

class SnowflakeCoreTrainer:
    """A class to manage the entire training and evaluation process."""
    def __init__(self, config: Config):
        self.config = config
        self.device, self.dtype = config.get_device_and_dtype()
        
        # 1. Tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size

        # 2. DataLoaders
        self.train_dataloader, self.val_dataloader = self._prepare_dataloaders()
        
        # 3. Model, Optimizer, and Scaler
        print("Initializing model...")
        self.model = SnowflakeCoreG1(
            vocab_size=self.vocab_size,
            embed_dim=config.EMBED_DIM,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            max_length=config.MAX_LENGTH,
            ffn_dim=config.FFN_DIM,
            dropout=config.DROPOUT
        ).to(self.device).to(self.dtype)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == torch.float16))

        # 4. Training state
        self.train_losses = []
        self.val_losses = []
        self.val_perplexities = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def _prepare_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Loads and splits the dataset, then creates DataLoaders."""
        print(f"Loading pre-training dataset: {self.config.DATASET_NAME}...")
        dataset = load_dataset(self.config.DATASET_NAME, split="train")
        full_dataset = TextDataset(dataset, self.tokenizer, self.config.MAX_LENGTH)
        
        val_size = int(self.config.VAL_SPLIT_RATIO * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        return train_dataloader, val_dataloader

    def _run_validation(self, max_batches: int = None) -> Tuple[float, float]:
        """
        Computes validation loss and perplexity.
        
        Args:
            max_batches (int, optional): Limits validation to a subset of batches.
                                         Defaults to None (full validation set).
        
        Returns:
            Tuple[float, float]: The average validation loss and perplexity.
        """
        self.model.eval()
        total_val_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for i, (input_ids, attention_mask, labels) in enumerate(self.val_dataloader):
                if max_batches and i >= max_batches:
                    break
                
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                with torch.cuda.amp.autocast(dtype=self.dtype, enabled=(self.dtype != torch.float32)):
                    logits = self.model(input_ids, attention_mask)
                    
                    shift_logits = logits[:, :-1].contiguous().view(-1, self.vocab_size)
                    shift_labels = labels[:, 1:].contiguous().view(-1)
                    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=self.tokenizer.pad_token_id, reduction='sum')
                
                non_pad_mask = (shift_labels != self.tokenizer.pad_token_id)
                num_tokens = non_pad_mask.sum().item()
                
                total_val_loss += loss.item()
                total_tokens += num_tokens
        
        avg_val_loss = total_val_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_val_loss)
        
        self.model.train()
        return avg_val_loss, perplexity

    def train(self):
        """The main training loop for the model."""
        print("Starting pre-training...")
        self.model.train()
        
        step_best_val_loss = float('inf')
        step_patience_counter = 0
        
        for epoch in range(self.config.EPOCHS):
            total_loss = 0
            total_tokens = 0
            self.optimizer.zero_grad()
            
            for step, (input_ids, attention_mask, labels) in enumerate(self.train_dataloader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                with torch.cuda.amp.autocast(dtype=self.dtype, enabled=(self.dtype != torch.float32)):
                    logits = self.model(input_ids, attention_mask)
                    
                    shift_logits = logits[:, :-1].contiguous().view(-1, self.vocab_size)
                    shift_labels = labels[:, 1:].contiguous().view(-1)
                    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=self.tokenizer.pad_token_id)

                non_pad_mask = (shift_labels != self.tokenizer.pad_token_id)
                num_tokens = non_pad_mask.sum().item()
                
                loss_scaled = loss / self.config.GRAD_ACCUM_STEPS
                self.scaler.scale(loss_scaled).backward()
                
                total_loss += loss.item() * num_tokens # De-normalize for logging
                total_tokens += num_tokens

                # Gradient accumulation and optimization step
                if (step + 1) % self.config.GRAD_ACCUM_STEPS == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                # Step-level validation and early stopping
                if (step + 1) % self.config.VALIDATE_EVERY_N_STEPS == 0:
                    step_val_loss, step_val_perplexity = self._run_validation(max_batches=10)
                    print(f"Epoch {epoch+1} | Step {step+1}: Quick Val Loss: {step_val_loss:.4f} | Val PPL: {step_val_perplexity:.2f}")

                    if step_val_loss < step_best_val_loss - self.config.STEP_EARLY_STOPPING_MIN_DELTA:
                        step_best_val_loss = step_val_loss
                        step_patience_counter = 0
                        self.best_model_state = self.model.state_dict().copy()
                        print(f"    New step-level best validation loss: {step_best_val_loss:.4f}")
                    else:
                        step_patience_counter += 1
                        print(f"    No step-level improvement for {step_patience_counter} checks")

                    if step_patience_counter >= self.config.STEP_EARLY_STOPPING_PATIENCE:
                        print(f"Step-level early stopping triggered at step {step+1}!")
                        if self.best_model_state:
                            self.model.load_state_dict(self.best_model_state)
                            print("Restored best step-level model state.")
                        return # End training
                
                if (step + 1) % 100 == 0:
                    avg_train_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
                    train_perplexity = math.exp(avg_train_loss)
                    print(f"Epoch {epoch+1} | Step {step+1}: Train Loss: {avg_train_loss:.4f} | Train PPL: {train_perplexity:.2f}")

            # End-of-epoch optimization step for remaining gradients
            if (len(self.train_dataloader)) % self.config.GRAD_ACCUM_STEPS != 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Epoch-level validation
            avg_train_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
            self.train_losses.append(avg_train_loss)
            
            val_loss, val_perplexity = self._run_validation()
            self.val_losses.append(val_loss)
            self.val_perplexities.append(val_perplexity)

            print(f"\n--- Epoch {epoch+1}/{self.config.EPOCHS} Summary ---")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train PPL: {math.exp(avg_train_loss):.2f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_perplexity:.2f}")

            # Epoch-level early stopping
            if val_loss < self.best_val_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                print(f"  New best validation loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"  No improvement for {self.patience_counter} epochs")

            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Epoch-level early stopping triggered after {epoch+1} epochs!")
                break
        
        # Restore best model state after training completes
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print("Restored best model state based on validation loss.")

    def save_model(self):
        """Saves the model in the Hugging Face format."""
        print(f"Saving model to {self.config.SAVE_PATH}")
        os.makedirs(self.config.SAVE_PATH, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(self.config.SAVE_PATH, "pytorch_model.bin")
        torch.save(self.model.to(torch.float32).state_dict(), model_path)
        
        # Save as safetensors if available
        try:
            from safetensors.torch import save_file as save_safetensors
            safetensors_path = os.path.join(self.config.SAVE_PATH, "model.safetensors")
            save_safetensors(self.model.to(torch.float32).state_dict(), safetensors_path)
            print(f"Model also saved as {safetensors_path}")
        except ImportError:
            print("safetensors not installed. To save as .safetensors, run: pip install safetensors")
            
        # Save model config
        actual_epochs = len(self.train_losses)
        config_dict = {
            "architectures": ["SnowflakeCoreG1"],
            "model_type": "snowflake_core",
            "vocab_size": self.vocab_size,
            "embed_dim": self.config.EMBED_DIM,
            "num_heads": self.config.NUM_HEADS,
            "num_layers": self.config.NUM_LAYERS,
            "max_length": self.config.MAX_LENGTH,
            "ffn_dim": self.config.FFN_DIM,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": getattr(self.tokenizer, 'bos_token_id', None),
            "unk_token_id": getattr(self.tokenizer, 'unk_token_id', None),
        }
        config_path = os.path.join(self.config.SAVE_PATH, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save training metrics
        metrics = {
            "epochs_completed": actual_epochs,
            "best_validation_loss": self.best_val_loss,
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_validation_loss": self.val_losses[-1] if self.val_losses else None,
            "training_history": {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "val_perplexities": self.val_perplexities,
            }
        }
        metrics_path = os.path.join(self.config.SAVE_PATH, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Save tokenizer files
        self.tokenizer.save_pretrained(self.config.SAVE_PATH)
        
        print(f"Model and tokenizer saved to {self.config.SAVE_PATH}/")

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
        """
        Generates text based on a given prompt.
        
        Args:
            prompt (str): The initial text to start generation from.
            max_new_tokens (int, optional): The maximum number of new tokens to generate.
            temperature (float, optional): The sampling temperature.
        
        Returns:
            str: The generated text.
        """
        self.model.eval()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Only use the last part of the sequence if it exceeds max_length
                if generated.size(1) > self.config.MAX_LENGTH:
                    generated = generated[:, -self.config.MAX_LENGTH:]

                with torch.cuda.amp.autocast(dtype=self.dtype, enabled=(self.dtype != torch.float32)):
                    logits = self.model(generated)
                
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits.float(), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

# ==============================================================================
# 4. Main Execution
# ==============================================================================

if __name__ == "__main__":
    config = Config()
    trainer = SnowflakeCoreTrainer(config)
    
    # Run the training process
    try:
        trainer.train()
        trainer.save_model()
    except KeyboardInterrupt:
        print("Training interrupted. Saving best model state...")
        trainer.save_model()
    
    # Generation Demo
    print("\n" + "="*20 + " Generation Demo " + "="*20 + "\n")
    prompt1 = "The history of the Roman Empire is"
    print(f"Prompt: {prompt1}")
    generated_text1 = trainer.generate(prompt1, max_new_tokens=100, temperature=0.8)
    print(f"Generated: {generated_text1}")
    print()
    
    prompt2 = "Photosynthesis is the process"
    print(f"Prompt: {prompt2}")
    generated_text2 = trainer.generate(prompt2, max_new_tokens=100, temperature=0.8)
    print(f"Generated: {generated_text2}")
