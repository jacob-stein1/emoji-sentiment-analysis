import pandas as pd
import re
from collections import Counter
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import extract_emojis

def summarize(csv_path):

    df = pd.read_csv(csv_path)
    
    # Get unique embedders
    table = {}
    text_embedders = df['text_embedder'].unique()
    emoji_embedders = df['emoji_embedder'].unique()


    for text_embedder in text_embedders:
        # Subdict for each text embedder
        table[text_embedder] = {}

        for emoji_embedder in emoji_embedders:
            # Make a sub dataframe for current pair
            subset = df[
                (df['text_embedder'] == text_embedder) & 
                (df['emoji_embedder'] == emoji_embedder)
            ]
            # Move on if empty
            if subset.empty:
                table[text_embedder][emoji_embedder] = "-"
                continue

            # Count emojis appearing in the misclassified texts
            emoji_counter = Counter()
            for text in subset['text']:
                emojis = extract_emojis(text)
                emoji_counter.update(emojis)
            
            # If no data move on
            if not emoji_counter:
                table[text_embedder][emoji_embedder] = "-"
            else:
                # Otherwise save most common
                most_common_emoji, count = emoji_counter.most_common(1)[0]
                table[text_embedder][emoji_embedder] = most_common_emoji
    
    tbl = pd.DataFrame.from_dict(table, orient='index')
    print(tbl)
    return tbl

if __name__ == "__main__":
    csv_path = "differing_predictions_logreg.csv"
    csv_path_neural = "differing_predictions_neural.csv"
    summarize(csv_path)
    summarize(csv_path_neural)
