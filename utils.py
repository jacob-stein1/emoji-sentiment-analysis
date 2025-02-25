import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

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