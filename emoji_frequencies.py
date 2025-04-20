import pandas as pd
import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from collections import Counter

random.seed(12345)

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
        for char in emoji:  # handles cases where multiple emojis are matched together
            emoji_counts[char] += 1

    return dict(emoji_counts)

df = pd.read_csv('training.csv')

sentiment_emoji_counts = { #was using Counter class, but replaced
    'positive': defaultdict(int),
    'negative': defaultdict(int),
    'neutral': defaultdict(int)
}

sentiment_emoji_counts_per_text = { #was using Counter class, but replaced
    'positive': defaultdict(int),
    'negative': defaultdict(int),
    'neutral': defaultdict(int)
}

# for index, row in df.iterrows():
#     sentiment = row['sentiment']
#     text = row['text']
#     emojis = extract_emoji_counts(text)
#     for (emoji,count) in emojis.items():
#         sentiment_emoji_counts[sentiment][emoji] += count
#         sentiment_emoji_counts_per_text[sentiment][emoji] += 1

def all_emojis(text):
    all_emojis = set()
    for index, row in text.iterrows():
        text = row['text']
        emojis = extract_emoji_counts(text)
        for (emoji,_) in emojis.items():
            all_emojis.add(emoji)
    return all_emojis

#all code below is designed to find the emoji that is evenly split between the different sentiments
# emoji_sentiments = {}
# for (sentiment,dic) in sentiment_emoji_counts_per_text.items():
#     for (emoji,count) in dic.items():
#         if emoji not in emoji_sentiments.keys():
#             emoji_sentiments[emoji] = [0]*3
#             #indexing goes, positive, negative, neutral
#         if sentiment == 'positive':
#             emoji_sentiments[emoji][0] = count
#         elif sentiment == 'negative':
#             emoji_sentiments[emoji][1] = count
#         else:
#             emoji_sentiments[emoji][2] = count


# for (emoji, counts) in emoji_sentiments.items():
#     thing = 0
#     for count in counts:
#         thing += count
#     for i in range(3):
#         emoji_sentiments[emoji][i] = emoji_sentiments[emoji][i]/thing



def sort_keys_by_proximity_to_equal_dist(input_dict):
    target = [0.333, 0.333, 0.333]

    def distance(vec):
        return sum(abs(a - b) for a, b in zip(vec, target))

    sorted_keys = sorted(input_dict.keys(), key=lambda k: distance(input_dict[k]))
    return sorted_keys


def find_lines_with_emoji(emoji,data):
    final_text = []
    sentiments = set()
    for index, row in data.iterrows():
        sentiment = row['sentiment']
        text = row['text']
        emojis = extract_emoji_counts(text)
        if emoji in emojis.keys():
            final_text.append((text,sentiment))
            sentiments.add(sentiment)
    if len(sentiments) == 3:
        return final_text
    return None
            
# print(find_lines_with_emoji('🤣',df))

def split_list_randomly(lst, split_ratio=0.8):
    lst_copy = lst[:]
    random.shuffle(lst_copy)
    split_index = int(len(lst_copy) * split_ratio)
    return lst_copy[:split_index], lst_copy[split_index:]

# lines = find_lines_with_emoji('🤣',df)
# print(len(lines))
# train,test = split_list_randomly(lines)
# print(len(train))
# print(test)


# print(sort_keys_by_proximity_to_equal_dist(emoji_sentiments))




# top_emojis = {
#     sentiment: counts.most_common(2) for sentiment, counts in sentiment_emoji_counts.items()
# }

# sentiments = list(top_emojis.keys())
# emoji_labels = [[emoji for emoji, _ in top_emojis[sentiment]] for sentiment in sentiments]
# emoji_counts = [[count for _, count in top_emojis[sentiment]] for sentiment in sentiments]

# fig, ax = plt.subplots(figsize=(10, 6))

# bar_width = 0.3
# x = np.arange(len(sentiments))

# for i in range(2):
#     ax.bar(
#         x + i * bar_width,
#         [emoji_counts[j][i] for j in range(len(sentiments))],
#         width=bar_width,
#         label=f'Emoji {i + 1}'
#     )

# ax.set_xlabel('Sentiment')
# ax.set_ylabel('Frequency')
# ax.set_title('Top 2 Emojis by Sentiment')
# ax.set_xticks(x + bar_width / 2)
# ax.set_xticklabels(sentiments)
# ax.legend()

# for i in range(2): 
#     for j in range(len(sentiments)):
#         ax.text(
#             x[j] + i * bar_width,
#             emoji_counts[j][i] + 10, 
#             emoji_labels[j][i],
#             ha='center',
#             va='bottom',
#             fontsize=16
#         )

# plt.tight_layout()
# plt.show()