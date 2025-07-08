import gensim.models
import pandas as pd
from scipy.stats import pearsonr
import os

# Load the pre-trained Word2Vec model
try:
    model = gensim.models.KeyedVectors.load_word2vec_format('reducedvector.bin', binary=True)
except FileNotFoundError:
    print("Error: reducedvector.bin not found in the current directory.")
    exit(1)

# Convert all words in the model to lowercase for consistency
model.key_to_index = {k.lower(): v for k, v in model.key_to_index.items()}

# Q1: Compute similarities for 'man' and 'woman' with 15 words
targets = ['man', 'woman']
words = ['wife', 'husband', 'child', 'queen', 'king', 'man', 'woman', 'birth', 
         'doctor', 'nurse', 'teacher', 'professor', 'engineer', 'scientist', 'president']

def compute_similarities(target, words, model):
    target = target.lower()
    data = []
    for word in words:
        word = word.lower()
        if word in model.key_to_index and target in model.key_to_index:
            sim = round(model.similarity(target, word), 2)
            data.append({'Word': word, 'Similarity': sim})
        else:
            data.append({'Word': word, 'Similarity': 'NA'})
    df = pd.DataFrame(data)
    df['Similarity'] = pd.to_numeric(df['Similarity'], errors='coerce')
    df = df.sort_values(by='Similarity', ascending=False, na_position='last')
    df = df.reset_index(drop=True)
    return df

# Generate Q1 outputs
df_man = compute_similarities('man', words, model)
df_man.to_csv('q1_man_similarities.csv', index=False)
print("Generated q1_man_similarities.csv")
df_woman = compute_similarities('woman', words, model)
df_woman.to_csv('q1_woman_similarities.csv', index=False)
print("Generated q1_woman_similarities.csv")

# Q2: BATS file analysis using E01 [country - capital].txt
try:
    with open('E01 [country - capital].txt', 'r') as f:
        pairs = [line.strip().split() for line in f if line.strip()]
except FileNotFoundError:
    print("Error: E01 [country - capital].txt not found in the current directory.")
    exit(1)

# Q2A: Compute similarity between capital and country
q2a_data = []
for capital, country in pairs:
    capital = capital.lower()
    country = country.lower()
    if capital in model.key_to_index and country in model.key_to_index:
        sim = round(model.similarity(capital, country), 2)
        q2a_data.append({'Capital': capital, 'Country': country, 'Similarity': sim})
    else:
        q2a_data.append({'Capital': capital, 'Country': country, 'Similarity': 'NA'})
df_q2a = pd.DataFrame(q2a_data)
df_q2a['Similarity'] = pd.to_numeric(df_q2a['Similarity'], errors='coerce')
df_q2a = df_q2a.sort_values(by='Similarity', ascending=False, na_position='last')
df_q2a.to_csv('q2a_similarities.csv', index=False)
print("Generated q2a_similarities.csv")

# Q2B: Bias analysis with race-related words
protected_words = ['black', 'white', 'asian']
q2b_data = []
for capital, country in pairs:
    capital = capital.lower()
    row = {'Capital': capital, 'Country': country.lower()}
    sims = []
    for pw in protected_words:
        pw = pw.lower()
        if capital in model.key_to_index and pw in model.key_to_index:
            sim = round(model.similarity(capital, pw), 2)
            row[f'Similarity_{pw}'] = sim
            sims.append(sim)
        else:
            row[f'Similarity_{pw}'] = 'NA'
            sims.append(None)
    if all(isinstance(s, (int, float)) for s in sims):
        diff = max(sims) - min(sims)
        row['Noticeable Difference'] = 'Yes' if diff > 0.1 else 'No'
    else:
        row['Noticeable Difference'] = 'NA'
    q2b_data.append(row)
df_q2b = pd.DataFrame(q2b_data)
df_q2b.to_csv('q2b_bias_analysis.csv', index=False)
print("Generated q2b_bias_analysis.csv")
print("\nQ2B Bias Observations:")
print("A 'noticeable difference' is defined as a difference greater than 0.1 between the highest and lowest similarity scores for 'black', 'white', and 'asian' in a row.")
print("Rows with 'Yes' in the 'Noticeable Difference' column indicate such differences.")

# Q3: Analogy completion
analogies = [
    ('king', 'throne', 'judge', None),
    ('giant', 'dwarf', 'genius', None),
    ('college', 'dean', 'jail', None),
    ('arc', 'circle', 'line', None),
    ('french', 'france', 'dutch', None),
    ('man', 'woman', 'king', None),
    ('water', 'ice', 'liquid', None),
    ('bad', 'good', 'sad', None),
    ('nurse', 'hospital', 'teacher', None),
    ('usa', 'pizza', 'japan', None),
    ('human', 'house', 'dog', None),
    ('grass', 'green', 'sky', None),
    ('video', 'cassette', 'computer', None),
    ('universe', 'planet', 'house', None),
    ('poverty', 'wealth', 'sickness', None)
]

manual_completions = [
    'court', 'idiot', 'warden', 'plane', 'netherlands', 'queen', 'solid', 
    'happy', 'school', 'sushi', 'kennel', 'blue', 'disk', 'room', 'health'
]

# Q3A: Manual analogy completion
q3a_data = []
for (a, b, c, _), w in zip(analogies, manual_completions):
    a, b, c, w = a.lower(), b.lower(), c.lower(), w.lower()
    if c in model.key_to_index and w in model.key_to_index:
        sim = round(model.similarity(c, w), 2)
        q3a_data.append({'Analogy': f"{a} is to {b} as {c} is to {w}", 'Similarity': sim})
    else:
        q3a_data.append({'Analogy': f"{a} is to {b} as {c} is to {w}", 'Similarity': 'NA'})
df_q3a = pd.DataFrame(q3a_data)
df_q3a.to_csv('q3a_manual_analogies.csv', index=False)
print("Generated q3a_manual_analogies.csv")

# Q3B: Model-generated analogy completion
q3b_data = []
for a, b, c, _ in analogies:
    a, b, c = a.lower(), b.lower(), c.lower()
    try:
        result = model.most_similar(positive=[c, b], negative=[a], topn=1)[0]
        w_model = result[0].lower()
        sim = round(model.similarity(c, w_model), 2)
        q3b_data.append({'Analogy': f"{a} is to {b} as {c} is to {w_model}", 'Similarity': sim})
    except KeyError:
        q3b_data.append({'Analogy': f"{a} is to {b} as {c} is to ?", 'Similarity': 'NA'})
df_q3b = pd.DataFrame(q3b_data)
df_q3b.to_csv('q3b_model_analogies.csv', index=False)
print("Generated q3b_model_analogies.csv")

# Q3C: Compute correlation
manual_sims = df_q3a['Similarity'].apply(lambda x: float(x) if x != 'NA' else None).dropna()
model_sims = df_q3b['Similarity'].apply(lambda x: float(x) if x != 'NA' else None).dropna()

if len(manual_sims) == len(model_sims) and len(manual_sims) > 1:
    correlation, _ = pearsonr(manual_sims, model_sims)
    print("\nQ3C Correlation Analysis:")
    print(f"Pearson correlation: {correlation:.3f}")
    if 0.00 <= abs(correlation) <= 0.19:
        strength = "very weak"
    elif 0.20 <= abs(correlation) <= 0.39:
        strength = "weak"
    elif 0.40 <= abs(correlation) <= 0.59:
        strength = "moderate"
    elif 0.60 <= abs(correlation) <= 0.79:
        strength = "strong"
    elif 0.80 <= abs(correlation) <= 1.00:
        strength = "very strong"
    print(f"Correlation strength: {strength}")
else:
    print("\nQ3C Correlation Analysis: Insufficient valid data for correlation.")