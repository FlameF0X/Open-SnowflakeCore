# OpenSnowflakeCore

**An open-source, educational implementation of a GPT-style transformer language model**

OpenSnowflakeCore is a clean, well-documented implementation of a decoder-only transformer architecture similar to GPT. This is an open-source version of an older iteration of the main SnowflakeCore codebase, designed for educational purposes and experimentation with transformer architectures.

## 🌟 Features

- **Compact Architecture**: ~10M parameter model designed for efficient training and experimentation
- **Modern Training Techniques**: Mixed precision training, gradient accumulation, and early stopping
- **Educational Focus**: Clean, well-commented code perfect for learning transformer architectures
- **Flexible Configuration**: Easy-to-modify hyperparameters and model settings
- **HuggingFace Integration**: Compatible tokenizers and standard dataset loading
- **Multi-level Early Stopping**: Both step-level and epoch-level early stopping for optimal training

## 🏗️ Architecture

The model implements a standard decoder-only transformer with the following components:

- **Fused Self-Attention**: Efficient QKV projection combining for better performance
- **Layer Normalization**: Pre-norm architecture (LayerNorm → Attention → LayerNorm → MLP)
- **Positional Embeddings**: Learned positional encodings up to max sequence length
- **Causal Masking**: Proper autoregressive attention masking
- **Dropout Regularization**: Configurable dropout for training stability

### Default Model Specifications

| Parameter | Value |
|-----------|-------|
| Parameters | ~10M |
| Embedding Dimension | 96 |
| Attention Heads | 4 |
| Transformer Layers | 8 |
| FFN Dimension | 384 |
| Max Sequence Length | 1024 |
| Vocabulary Size | GPT-2 tokenizer (~50K) |

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers datasets safetensors
```

### Basic Usage

```python
from your_script import Config, SnowflakeCoreTrainer

# Initialize with default configuration
config = Config()
trainer = SnowflakeCoreTrainer(config)

# Start training
trainer.train()

# Save the trained model
trainer.save_model()

# Generate text
generated_text = trainer.generate("The future of AI is", max_new_tokens=50)
print(generated_text)
```

### Custom Configuration

```python
# Modify configuration for your needs
config = Config()
config.EMBED_DIM = 128        # Larger model
config.NUM_LAYERS = 12        # Deeper model
config.BATCH_SIZE = 4         # Larger batches
config.LEARNING_RATE = 1e-4   # Different learning rate
config.MAX_LENGTH = 512       # Shorter sequences

trainer = SnowflakeCoreTrainer(config)
```

## ⚙️ Configuration Options

### Model Architecture
- `EMBED_DIM`: Embedding dimension (default: 96)
- `NUM_HEADS`: Number of attention heads (default: 4)
- `NUM_LAYERS`: Number of transformer blocks (default: 8)
- `FFN_DIM`: Feed-forward network dimension (default: 384)
- `MAX_LENGTH`: Maximum sequence length (default: 1024)
- `DROPOUT`: Dropout probability (default: 0.1)

### Training Parameters
- `BATCH_SIZE`: Training batch size (default: 1)
- `EPOCHS`: Maximum training epochs (default: 5)
- `LEARNING_RATE`: AdamW learning rate (default: 2e-4)
- `GRAD_ACCUM_STEPS`: Gradient accumulation steps (default: 32)

### Early Stopping
- `EARLY_STOPPING_PATIENCE`: Epoch-level patience (default: 1)
- `STEP_EARLY_STOPPING_PATIENCE`: Step-level patience (default: 5)
- `VALIDATE_EVERY_N_STEPS`: Validation frequency (default: 500)

### Dataset
- `DATASET_NAME`: HuggingFace dataset name (default: "common-pile/wikimedia_filtered")
- `VAL_SPLIT_RATIO`: Validation split ratio (default: 0.05)

## 🔧 Hardware Requirements

### Minimum Requirements
- **GPU**: 4GB VRAM (with batch_size=1)
- **RAM**: 8GB system RAM
- **Storage**: 5GB free space

### Recommended Requirements
- **GPU**: 8GB+ VRAM (RTX 3070/4060 or better)
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space (for larger datasets)

### Automatic Precision Selection
The model automatically selects the best precision for your hardware:
- **BF16**: On supported modern GPUs (RTX 30/40 series, A100, etc.)
- **FP16**: On older CUDA GPUs
- **FP32**: On CPU or unsupported GPUs

## 📊 Training Features

### Mixed Precision Training
Automatic mixed precision with proper gradient scaling for faster training and reduced memory usage.

### Gradient Accumulation
Simulate larger batch sizes by accumulating gradients over multiple steps.

### Multi-level Early Stopping
- **Step-level**: Quick validation every N steps with early stopping
- **Epoch-level**: Full validation at epoch end with early stopping

### Comprehensive Logging
- Training loss and perplexity tracking
- Validation metrics monitoring
- Automatic best model state preservation

## 💾 Model Saving

The trained model is saved in HuggingFace-compatible format:

```
SnowflakeCore-G1-Small/
├── config.json              # Model configuration
├── pytorch_model.bin        # PyTorch weights
├── model.safetensors        # SafeTensors format (if available)
├── training_metrics.json    # Training history and metrics
├── tokenizer.json           # Tokenizer configuration
├── tokenizer_config.json    # Tokenizer settings
└── vocab.json               # Vocabulary
```

## 🎯 Use Cases

### Educational
- Learn transformer architecture implementation
- Experiment with different hyperparameters
- Understand attention mechanisms and training dynamics

### Research
- Baseline model for architecture experiments
- Quick prototyping of new ideas
- Ablation studies on transformer components

### Small-scale Applications
- Text completion for specific domains
- Fine-tuning for specialized tasks
- Resource-constrained deployment scenarios

## ⚠️ Important Notes

### Dataset Warning
The default dataset `"common-pile/wikimedia_filtered"` is large and requires significant RAM and storage. For quick experimentation, consider using smaller datasets like:
- `"wikitext-2-raw-v1"`
- `"wikitext-103-raw-v1"`
- Custom text datasets

### Memory Considerations
- Large datasets may require substantial RAM for loading
- Consider using streaming datasets for very large corpora
- Monitor GPU memory usage during training

## 🔄 Version Information

This is an **open-source version of an older iteration** of the main SnowflakeCore codebase. It represents a simplified, educational implementation that captures the core concepts while being accessible for learning and experimentation.

### Differences from Main Version
- Simplified architecture for educational clarity
- Reduced complexity in training pipeline
- Focus on readability over maximum performance
- Smaller default model size for accessibility

## 🤝 Contributing

This educational codebase welcomes contributions! Areas for improvement:
- Additional architectural variants
- More efficient training techniques
- Better documentation and examples
- Support for different tokenizers
- Streaming dataset support


## 🙏 Acknowledgments

- Inspired by the GPT family of models
- Built with PyTorch and HuggingFace Transformers
- Educational implementation for the community

---

**Note**: This is an educational implementation. For production use cases, consider using established libraries like HuggingFace Transformers or other optimized frameworks.
