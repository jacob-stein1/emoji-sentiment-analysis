import pandas as pd
import re
from collections import Counter

def extract_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F700-\U0001F77F"  # alchemical
        "\U0001F780-\U0001F7FF"  # geometric shapes extended
        "\U0001F800-\U0001F8FF"  # supplemental arrows
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.findall(str(text))

def summarize_most_misclassified_emoji(csv_path):
    df = pd.read_csv(csv_path)
    
    table = {}

    text_embedders = df['text_embedder'].unique()
    emoji_embedders = df['emoji_embedder'].unique()

    for text_embedder in text_embedders:
        table[text_embedder] = {}
        for emoji_embedder in emoji_embedders:
            subset = df[(df['text_embedder'] == text_embedder) & (df['emoji_embedder'] == emoji_embedder)]
            
            if subset.empty:
                table[text_embedder][emoji_embedder] = "-"
                continue

            # Count emojis appearing in the misclassified texts
            emoji_counter = Counter()
            for text in subset['text']:
                emojis = extract_emojis(text)
                emoji_counter.update(emojis)
            
            if not emoji_counter:
                table[text_embedder][emoji_embedder] = "-"
            else:
                most_common_emoji, count = emoji_counter.most_common(1)[0]
                table[text_embedder][emoji_embedder] = most_common_emoji
    
    pivot = pd.DataFrame.from_dict(table, orient='index')
    print(pivot)
    return pivot

if __name__ == "__main__":
    # Path to your differing predictions file
    csv_path = "differing_predictions_logreg.csv"
    csv_path_neural = "differing_predictions_neural.csv"
    summarize_most_misclassified_emoji(csv_path)
    summarize_most_misclassified_emoji(csv_path_neural)
