import pandas as pd
import numpy as np
import re
import random
from collections import defaultdict
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import emoji_pattern

random.seed(12345)

def extract_emoji_counts(text):
    emoji_counts = defaultdict(int)
    for emoji in emoji_pattern().findall(str(text)):
        for char in emoji:
            emoji_counts[char] += 1
    return dict(emoji_counts)

def populate(dataframe):
    total_count = {
        'positive': defaultdict(int),
        'negative': defaultdict(int),
        'neutral': defaultdict(int)
    }

    per_text_count = {
        'positive': defaultdict(int),
        'negative': defaultdict(int),
        'neutral': defaultdict(int)
    }

    for index, row in dataframe.iterrows():
        sentiment = row['sentiment']
        text = row['text']
        emojis = extract_emoji_counts(text)
        for emoji, count in emojis.items():
            total_count[sentiment][emoji] += count
            per_text_count[sentiment][emoji] += 1

    return total_count, per_text_count

def find_balanced(counts, output_file):
    emoji_sentiments = {}

    for sentiment, dic in counts.items():
        for emoji, count in dic.items():
            if emoji not in emoji_sentiments:
                emoji_sentiments[emoji] = [0, 0, 0]
            if sentiment == 'positive':
                emoji_sentiments[emoji][0] = count
            elif sentiment == 'negative':
                emoji_sentiments[emoji][1] = count
            elif sentiment == 'neutral':
                emoji_sentiments[emoji][2] = count

    rows = []
    for emoji, counts in emoji_sentiments.items():
        total = sum(counts)
        if total > 0:
            normalized = [c / total for c in counts]
            rows.append([emoji] + counts + normalized)

    df = pd.DataFrame(rows, columns=['emoji', 'pos_count', 'neg_count', 'neutral_count', 'pos_norm', 'neg_norm', 'neutral_norm'])
    target = np.array([0.333, 0.333, 0.333])
    df['distance'] = df[['pos_norm', 'neg_norm', 'neutral_norm']].apply(lambda x: np.sum(np.abs(x - target)), axis=1)
    df = df.sort_values(by='distance')
    df.to_csv(output_file, index=False)

def emoji_lines(emoji, dataframe):
    final_texts = []
    sentiments_found = set()

    for index, row in dataframe.iterrows():
        sentiment = row['sentiment']
        text = row['text']
        emojis = extract_emoji_counts(text)
        if emoji in emojis:
            final_texts.append((text, sentiment))
            sentiments_found.add(sentiment)

    if len(sentiments_found) == 3:
        return final_texts
    return None

if __name__ == "__main__":
    df = pd.read_csv('../training.csv')

    counts, counts_per_text = populate(df)

    find_balanced(counts, 'balanced_total_counts.csv')
    find_balanced(counts_per_text, 'balanced_per_text_counts.csv')
