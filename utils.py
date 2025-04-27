import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

# From https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1?permalink_comment_id=3208085
def emoji_pattern():
    pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF" 
        "\U0001F680-\U0001F6FF"  
        "\U0001F700-\U0001F77F"  
        "\U0001F780-\U0001F7FF"  
        "\U0001F800-\U0001F8FF"  
        "\U0001F900-\U0001F9FF"  
        "\U0001FA00-\U0001FA6F" 
        "\U0001FA70-\U0001FAFF"  
        "\U00002702-\U000027B0" 
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE
    )
    return pattern

def extract_emojis(text):
    return emoji_pattern().findall(str(text))

def remove_emojis(text):
    return emoji_pattern().sub(r"", str(text))
