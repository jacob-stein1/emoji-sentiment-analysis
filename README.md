## Project Description

This project tests various embedding techniques for sentiment classification on text containing emojis. We use one embedder for text and one for emojis, then pass both embeddings into one classifier, and only text embeddings into another. We measure the difference in accuracy as the impact attributable to the emoji embeddings. We do this for a logistic regression and neural classifier.

## Project Modules

### Text Embedders (`text_embedders/`)
- `base.py`: Abstract base class for text embedders
- `bert.py`: BERT text embedding with GPU support
- `doc2vec.py`: Doc2Vec for text embedding
- `fasttext.py`: FastText for text embedding
- `tfidf.py`: TF-IDF text embedding
- `word2vec.py`: Word2Vec for text embedding

### Emoji Embedders (`emoji_embedder/`)
- `base.py`: Abstract base class for emoji embedders
- `bert.py`: BERT emoji embedding with GPU support
- `doc2vec.py`: Doc2Vec for emoji embedding
- `fasttext.py`: FastText for emoji embedding
- `tfidf.py`: TF-IDF emoji embedding
- `word2vec.py`: Word2Vec for emoji embedding

### Evaluation (`evaluation/`)
- `embedder_evaluation.py`: Evaluates different combinations of text and emoji embedders
- `neural_embedder_evaluation.py`: Neural network-based evaluation of embedder combinations
- `misclassified_table.py`: Prints a table of most misclassified emoji by the text-only classifier
- `neural_models.py`: Contains neural model architecture
- `print_pmfs.py`: Prints out table of sentiment classes of most common emojis

### Utils (`utils.py`)
- Contains functions for emoji pattern matching and text preprocessing

### Dataset Processing (`combining_datasets/`)
- `compile.py`: Scripts for combining and preprocessing different sentiment datasets

### Emoji Analysis (`emoji_frequency_analysis/`)
- `emoji_frequencies.py`: Counts usage and frequency of emojis in CSVs

### Old Proof of Concept (`testing.py`)
- Runs a basic test with TF-IDF text embedding, adn one hot emoji embedding

### Dataset (`training.csv`)
- Unified dataset, reproducible by running `compile.py`

## Reproducibility  

1. Clone the repository and install dependencies
```bash
pip install -r requirements.txt
```
2. Run `embedder_evaluation.py` and `neural_embedder_evaluation.py` to test embedding combinations
3. You can produce the figures in our report with the following scripts:
- Table 1: `embedder_evaluation.py`
- Table 2: `misclassified_table.py`
- Table 3: `neural_embedder_evaluation.py`
- Table 4: `misclassified_table.py`
- Table 5: `print_pmfs.py`
- Table 6: `print_pmfs.py`
- Table 7: Novak et al.
- Figure 1: `neural_models.py`
- Figure 2: `neural_embedder_evaluation.py`

Note that running `neural_embedder_evaluation.py` takes around 15 minutes with a GPU, and several hours without one. 
