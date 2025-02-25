import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter

# Function to extract emojis from text
def extract_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.findall(str(text))

df = pd.read_csv('training.csv')

sentiment_emoji_counts = {
    'positive': Counter(),
    'negative': Counter(),
    'neutral': Counter()
}

for index, row in df.iterrows():
    sentiment = row['sentiment']
    text = row['text']
    emojis = extract_emojis(text)
    sentiment_emoji_counts[sentiment].update(emojis)

top_emojis = {
    sentiment: counts.most_common(2) for sentiment, counts in sentiment_emoji_counts.items()
}

sentiments = list(top_emojis.keys())
emoji_labels = [[emoji for emoji, _ in top_emojis[sentiment]] for sentiment in sentiments]
emoji_counts = [[count for _, count in top_emojis[sentiment]] for sentiment in sentiments]

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.3
x = np.arange(len(sentiments))

for i in range(2):
    ax.bar(
        x + i * bar_width,
        [emoji_counts[j][i] for j in range(len(sentiments))],
        width=bar_width,
        label=f'Emoji {i + 1}'
    )

ax.set_xlabel('Sentiment')
ax.set_ylabel('Frequency')
ax.set_title('Top 2 Emojis by Sentiment')
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(sentiments)
ax.legend()

for i in range(2): 
    for j in range(len(sentiments)):
        ax.text(
            x[j] + i * bar_width,
            emoji_counts[j][i] + 10, 
            emoji_labels[j][i],
            ha='center',
            va='bottom',
            fontsize=16
        )

plt.tight_layout()
plt.show()