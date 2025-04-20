import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

def extract_emojis(text):
    emoji_pattern = re.compile(
        "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F" "\U0001F780-\U0001F7FF" "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF" "\U0001FA00-\U0001FA6F" "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0" "\U000024C2-\U0001F251" "]+", flags=re.UNICODE
    )
    return emoji_pattern.findall(str(text))

def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F" "\U0001F780-\U0001F7FF" "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF" "\U0001FA00-\U0001FA6F" "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0" "\U000024C2-\U0001F251" "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r"", str(text))

def analyze_pmf_difference(pmf_df):

    pmf_with_emoji = pmf_df["with_emoji"].values
    pmf_without_emoji = pmf_df["without_emoji"].values

    kl_divergence = scipy.stats.entropy(pmf_with_emoji, pmf_without_emoji)
    cosine_similarity = np.dot(pmf_with_emoji, pmf_without_emoji) / (
        np.linalg.norm(pmf_with_emoji) * np.linalg.norm(pmf_without_emoji)
    )
    euclidean_distance = np.linalg.norm(pmf_with_emoji - pmf_without_emoji)

    print(f"KL Divergence: {kl_divergence:.4f}")
    print(f"Cosine Similarity: {cosine_similarity:.4f}")
    print(f"Euclidean Distance: {euclidean_distance:.4f}")

    plt.figure(figsize=(8, 5))
    width = 0.4 
    labels = pmf_df.index

    plt.bar(np.arange(len(labels)) - width/2, pmf_with_emoji, width=width, label="With Emoji", alpha=0.7)
    plt.bar(np.arange(len(labels)) + width/2, pmf_without_emoji, width=width, label="Without Emoji", alpha=0.7)

    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.ylabel("Probability")
    plt.title("Effect of Emoji on Sentiment PMFs")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

    return kl_divergence, cosine_similarity, euclidean_distance

def analyze_batch_pmf_differences(texts, classifier_with_emoji, classifier_without_emoji, 
                                text_embedder, emoji_embedder, label_encoder):
    all_metrics = {
        'kl_divergences': [],
        'cosine_similarities': [],
        'euclidean_distances': [],
        'individual_pmfs': []
    }
    
    for text in texts:
        # Get text embeddings
        text_embedding = text_embedder.transform([text]).toarray()
        emoji_embedding = emoji_embedder.transform([text])
        
        # Combine features for emoji classifier
        combined_embedding = np.hstack((text_embedding, emoji_embedding))
        
        # Get PMFs from both classifiers
        pmf_with_emoji = classifier_with_emoji.predict_proba(combined_embedding)[0]
        pmf_without_emoji = classifier_without_emoji.predict_proba(text_embedding)[0]
        
        # Create DataFrame for this sample
        pmf_df = pd.DataFrame({
            "with_emoji": pmf_with_emoji,
            "without_emoji": pmf_without_emoji
        }, index=label_encoder.classes_)
        
        # Calculate metrics for this sample
        kl_div = scipy.stats.entropy(pmf_with_emoji, pmf_without_emoji)
        cos_sim = np.dot(pmf_with_emoji, pmf_without_emoji) / (
            np.linalg.norm(pmf_with_emoji) * np.linalg.norm(pmf_without_emoji)
        )
        euc_dist = np.linalg.norm(pmf_with_emoji - pmf_without_emoji)
        
        # Store metrics
        all_metrics['kl_divergences'].append(kl_div)
        all_metrics['cosine_similarities'].append(cos_sim)
        all_metrics['euclidean_distances'].append(euc_dist)
        all_metrics['individual_pmfs'].append(pmf_df)
    
    aggregate_stats = {
        'mean_kl_divergence': np.mean(all_metrics['kl_divergences']),
        'std_kl_divergence': np.std(all_metrics['kl_divergences']),
        'mean_cosine_similarity': np.mean(all_metrics['cosine_similarities']),
        'std_cosine_similarity': np.std(all_metrics['cosine_similarities']),
        'mean_euclidean_distance': np.mean(all_metrics['euclidean_distances']),
        'std_euclidean_distance': np.std(all_metrics['euclidean_distances'])
    }
    
    return {
        'aggregate_stats': aggregate_stats,
        'individual_metrics': all_metrics
<<<<<<< HEAD
    }
=======
    }
>>>>>>> fb142b9 (Generalized)
