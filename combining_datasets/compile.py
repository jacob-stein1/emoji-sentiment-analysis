import pandas as pd
import re
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import emoji_pattern

def contains_emoji(text):
    return bool(emoji_pattern().search(str(text)))

# Parse pos.csv, neg.csv, and posneg.csv
def parse_csv(file_path):
    df = pd.read_csv(file_path)
    df = df[['sentiment', 'post']]
    df.columns = ['sentiment', 'text']
    return df
# Parse sentiment-tweets-1.csv
def parse_tweets_1(file_path):
    df = pd.read_csv(file_path)
    df = df[['sentiment', 'Text']]
    df.columns = ['sentiment', 'text']
    return df
# Parse sentiment-tweets-2.csv
def parse_tweets_2(file_path):
    df = pd.read_csv(file_path, header=None)
    df = df[[2, 3]] 
    df.columns = ['sentiment', 'text']
    print(df)
    return df

# Produce DF
pos_df = parse_csv('pos.csv')
neg_df = parse_csv('neg.csv')
posneg_df = parse_csv('posneg.csv')
tweets_1_df = parse_tweets_1('sentiment-tweets-1.csv')
tweets_2_df = parse_tweets_2('sentiment-tweets-2.csv')
combined_df = pd.concat([pos_df, neg_df, posneg_df, tweets_1_df, tweets_2_df], ignore_index=True)

# Unify labels
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