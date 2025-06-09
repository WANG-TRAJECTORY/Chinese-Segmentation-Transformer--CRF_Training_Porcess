# Chinese Word Segmentation with Transformer-CRF

This project implements a Chinese word segmentation system using a Transformer encoder with Conditional Random Field (CRF) for sequence labeling. The model follows the BMES (Begin, Middle, End, Single) tagging scheme for word segmentation.

## Model Architecture

- **Transformer Encoder**: 4 layers with 8 attention heads and 1024 hidden dimension
- **Embedding Dimension**: 256
- **CRF Layer**: Handles transition constraints between tags
- **Positional Encoding**: Added to capture sequence order information
- **Tagging Scheme**: BMESO (Begin, Middle, End, Single, Padding)

## Dataset Format

The dataset should be in JSON format with each line containing:
```json
{"text": "word1 word2 ...", "label": "tag1 tag2 ..."}
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- Transformers library
- scikit-learn
- tqdm
- matplotlib

Install dependencies:
```bash
pip install torch transformers scikit-learn tqdm matplotlib
```

