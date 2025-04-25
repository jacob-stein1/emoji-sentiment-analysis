import pandas as pd
import numpy as np
import re
import random
from collections import defaultdict

random.seed(12345)

def extract_emoji_counts(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map
        "\U0001F700-\U0001F77F"  # Alchemical
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    emoji_counts = defaultdict(int)
    for emoji in emoji_pattern.findall(str(text)):
        for char in emoji:
            emoji_counts[char] += 1
    return dict(emoji_counts)

def populate_sentiment_emoji_counts(dataframe):
    sentiment_emoji_counts = {
        'positive': defaultdict(int),
        'negative': defaultdict(int),
        'neutral': defaultdict(int)
    }

    sentiment_emoji_counts_per_text = {
        'positive': defaultdict(int),
        'negative': defaultdict(int),
        'neutral': defaultdict(int)
    }

    for index, row in dataframe.iterrows():
        sentiment = row['sentiment']
        text = row['text']
        emojis = extract_emoji_counts(text)
        for emoji, count in emojis.items():
            sentiment_emoji_counts[sentiment][emoji] += count
            sentiment_emoji_counts_per_text[sentiment][emoji] += 1

    return sentiment_emoji_counts, sentiment_emoji_counts_per_text

def find_most_balanced_emojis_and_save(sentiment_emoji_counts, output_file):
    emoji_sentiments = {}

    for sentiment, dic in sentiment_emoji_counts.items():
        for emoji, count in dic.items():
            if emoji not in emoji_sentiments:
                emoji_sentiments[emoji] = [0, 0, 0]  # [positive, negative, neutral]
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

def find_lines_with_emoji(emoji, dataframe):
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

    sentiment_emoji_counts, sentiment_emoji_counts_per_text = populate_sentiment_emoji_counts(df)

    find_most_balanced_emojis_and_save(sentiment_emoji_counts, 'balanced_total_counts.csv')
    find_most_balanced_emojis_and_save(sentiment_emoji_counts_per_text, 'balanced_per_text_counts.csv')

    lines = find_lines_with_emoji('🤣', df)
    if lines:
        print(f"\nTotal lines with '🤣': {len(lines)}")
        random.shuffle(lines)
        split_index = int(0.8 * len(lines))
        train_lines = lines[:split_index]
        test_lines = lines[split_index:]
        print(f"Train: {len(train_lines)}, Test: {len(test_lines)}")
