# Emoji Sentiment Analysis

This project implements a comprehensive sentiment analysis system that combines text and emoji embeddings to improve sentiment classification accuracy. The system uses various embedding techniques for both text and emoji analysis, with GPU support for deep learning models.

## Project Structure

### Text Embedders (`text_embedders/`)
- `base.py`: Abstract base class for text embedders
- `bert.py`: BERT-based text embedding with GPU support
- `doc2vec.py`: Doc2Vec implementation for text embedding
- `fasttext.py`: FastText implementation for text embedding
- `tfidf.py`: TF-IDF based text embedding
- `word2vec.py`: Word2Vec implementation for text embedding

### Emoji Embedders (`emoji_embedder/`)
- `base.py`: Abstract base class for emoji embedders
- `bert.py`: BERT-based emoji embedding with GPU support
- `doc2vec.py`: Doc2Vec implementation for emoji embedding
- `fasttext.py`: FastText implementation for emoji embedding
- `tfidf.py`: TF-IDF based emoji embedding
- `word2vec.py`: Word2Vec implementation for emoji embedding

### Evaluation (`evaluation/`)
- `embedder_evaluation.py`: Evaluates different combinations of text and emoji embedders
- `neural_embedder_evaluation.py`: Neural network-based evaluation of embedder combinations

### Utils (`utils.py`)
- Contains utility functions for emoji pattern matching and text preprocessing
- Provides functions for analyzing probability mass function differences

### Dataset Processing (`combining_datasets/`)
- `compile.py`: Scripts for combining and preprocessing different sentiment datasets

### Emoji Analysis (`emoji_frequency_analysis/`)
- `emoji_frequencies.py`: Analyzes emoji usage patterns and frequencies in the dataset

## Features

- Multiple embedding techniques for both text and emoji analysis
- GPU acceleration support for BERT models
- Hybrid approach combining text and emoji embeddings
- Comprehensive evaluation framework
- Support for various deep learning architectures
- Dataset preprocessing and combination utilities

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in CSV format with 'text' and 'sentiment' columns
2. Choose appropriate text and emoji embedders
3. Run evaluation scripts to compare different embedding combinations

Example:
```python
from text_embedders.bert import BERTEmbedder
from emoji_embedder.bert import BERTEmojiEmbedder

# Initialize embedders
text_embedder = BERTEmbedder()
emoji_embedder = BERTEmojiEmbedder()

# Fit and transform
text_embeddings = text_embedder.fit_transform(texts)
emoji_embeddings = emoji_embedder.fit_transform(texts)
```

## GPU Support

The BERT-based embedders automatically detect and utilize GPU if available. No additional configuration is needed.

## Contributing

Feel free to submit issues and enhancement requests! 