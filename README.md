# Open-Snowflake-G0

Open-Snowflake-G0 is a open-sourse pre-train version of Snowflake-G0-Release series.

This is the initial release of the Snowflake (Snowflake-G0-Release) series language models, trained on the DialogMLM-50K dataset with optimized memory usage.

## Model details
- Architecture: SnowflakeCore
- Hidden size: {dm}
- Number of attention heads: {nh}
- Number of layers: {nl}
- Feed-forward dimension: {fd}
- Maximum sequence length: {msl}
- Vocabulary size: {vs}

## HuggingFace Transformers Compatibility
This model is fully compatible with the HuggingFace Transformers library. You can load it using:

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path/to/snowflake_g0_release")
config = AutoConfig.from_pretrained("path/to/snowflake_g0_release")
model = AutoModel.from_pretrained("path/to/snowflake_g0_release")
```

## Memory Optimization Techniques
- Mixed precision training
- Gradient accumulation ({gas} steps)
- Fused QKV projection
- Pre-norm architecture
- Weight tying between embedding and output layers
- Half-precision model storage

The model weights are stored in both PyTorch (.bin) and safetensors format for improved security, loading efficiency, and compatibility.
