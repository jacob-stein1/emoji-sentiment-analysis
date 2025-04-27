import pandas as pd

def main():

    df = pd.read_csv('neural_pmfs/BERT_Word2Vec_pmf.csv')
    
    # Drop nulls, empty, and NaNs, compute total
    df = df.dropna(subset=['emoji'])                 
    df = df[df['emoji'].str.strip() != '']            
    df['total'] = df['pos_count'] + df['neg_count'] + df['neutral_count']    
    df_sorted = df.sort_values(by='total', ascending=False)
    
    # Normalization
    df_normalized = df_sorted.copy()
    df_normalized['pos_norm'] = df_sorted['pos_count'] / df_sorted['total']
    df_normalized['neg_norm'] = df_sorted['neg_count'] / df_sorted['total']
    df_normalized['neutral_norm'] = df_sorted['neutral_count'] / df_sorted['total']
    
    # Select and print columns
    df_normalized = df_normalized[['emoji', 'pos_norm', 'neg_norm', 'neutral_norm', 'total']]    
    print(df_normalized.head(10))

if __name__ == "__main__":
    main()
