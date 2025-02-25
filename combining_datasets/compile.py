import pandas as pd
import re

# Function to check if a string contains an emoji
def contains_emoji(text):
    if pd.isna(text):  # Handle NaN values
        return False
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
    return bool(emoji_pattern.search(str(text)))

# Function to parse pos.csv, neg.csv, and posneg.csv
def parse_simple_csv(file_path):
    df = pd.read_csv(file_path)
    df = df[['sentiment', 'post']]
    df.columns = ['sentiment', 'text']
    return df

# Function to parse sentiment-tweets-1.csv
def parse_tweets_1(file_path):
    df = pd.read_csv(file_path)
    df = df[['sentiment', 'Text']]
    df.columns = ['sentiment', 'text']
    return df

# Function to parse sentiment-tweets-2.csv
def parse_tweets_2(file_path):
    df = pd.read_csv(file_path, header=None)
    df = df[[2, 3]] 
    df.columns = ['sentiment', 'text']
    print(df)
    return df

# Parse all files
pos_df = parse_simple_csv('pos.csv')
neg_df = parse_simple_csv('neg.csv')
posneg_df = parse_simple_csv('posneg.csv')
tweets_1_df = parse_tweets_1('sentiment-tweets-1.csv')
tweets_2_df = parse_tweets_2('sentiment-tweets-2.csv')

# Combine all dataframes
combined_df = pd.concat([pos_df, neg_df, posneg_df, tweets_1_df, tweets_2_df], ignore_index=True)

# Unify sentiment labels
def unify_sentiment(sentiment):
    if isinstance(sentiment, str):
        sentiment = sentiment.lower().strip()
        if sentiment in ['positive', 'pos', '1']:
            return 'positive'
        elif sentiment in ['negative', 'neg', '-1']:
            return 'negative'
        elif sentiment in ['neutral', 'neu', '0']:
            return 'neutral'
    return 'neutral'

combined_df['sentiment'] = combined_df['sentiment'].apply(unify_sentiment)

combined_df = combined_df[combined_df['text'].apply(contains_emoji)]

combined_df = combined_df.drop_duplicates(subset=['text'])

combined_df.to_csv('training.csv', index=False)

print("Combined CSV saved as 'combined_emoji_sentiment.csv'")