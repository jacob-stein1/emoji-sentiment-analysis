import pandas as pd

# Load your CSV
df = pd.read_csv('kral_novak.csv')

# Mapping from the emoji latex names to their Unicode name in the CSV
emoji_names = {
    'anguished': 'ANGUISHED FACE',
    'blush': 'SMILING FACE WITH SMILING EYES',
    'rofl': 'ROLLING ON THE FLOOR LAUGHING',
    'joy': 'FACE WITH TEARS OF JOY',
    'sob': 'LOUDLY CRYING FACE',
    'person-facepalming': 'FACE PALM',
    'face-with-symbols-on-mouth': 'FACE WITH SYMBOLS ON MOUTH',
    'heart': 'HEAVY BLACK HEART',  # Unicode name is usually "HEAVY BLACK HEART"
    'pensive': 'PENSIVE FACE',
    'fire': 'FIRE'
}

# Go through each desired emoji
for latex_name, unicode_name in emoji_names.items():
    row = df[df['Unicode name'] == unicode_name]
    
    if not row.empty:
        neg = row['Negative'].values[0]
        neu = row['Neutral'].values[0]
        pos = row['Positive'].values[0]
        total = neg + neu + pos

        neg_norm = neg / total
        neu_norm = neu / total
        pos_norm = pos / total

        print(f"\\emoji{{{latex_name}}} & {neg_norm:.3f} & {neu_norm:.3f} & {pos_norm:.3f} \\\\")
    else:
        print(f"\\emoji{{{latex_name}}} & NOT FOUND \\\\")
