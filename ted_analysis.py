"""
TED Talk Linguistic Analysis Project
Author: Osaretin Igbinoba
Email: igbi1963@mylaurier.ca
__updated__ = "2025-10-11"
"""

import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ast
from tqdm import tqdm
import re
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.6f}'.format)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded")
except OSError:
    print("spaCy model not found, using blank model")
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

print("\n" + "="*70)
print("TED TALK ANALYSIS")
print("="*70 + "\n")

print("Loading data...")
df_main = pd.read_csv('ted_main.csv')
df_transcripts = pd.read_csv('transcripts.csv')
print(f"Loaded {df_main.shape[0]} talks from ted_main.csv")
print(f"Loaded {df_transcripts.shape[0]} transcripts\n")

print("Dataset info:")
print(df_main.head())
print("\n")
print(df_main.info())
print("\n")

print(df_transcripts.head())
print("\n")
print(df_transcripts.info())
print("\n")

print("Merging datasets...")
df_ted = pd.merge(df_main, df_transcripts, on='url', how='inner')
print(f"Merged dataset has {df_ted.shape[0]} rows\n")

print("Checking for missing data...")
missing = df_ted.isnull().sum()
print(missing[missing > 0])
print()

df_ted = df_ted.dropna(subset=['transcript'])
print(f"Final dataset: {df_ted.shape[0]} talks\n")

print("="*70)
print("SECTION 2 COMPLETE")
print("="*70 + "\n")

print("Parsing ratings...")
df_ted['ratings_parsed'] = df_ted['ratings'].apply(ast.literal_eval)
print("Ratings parsed\n")

def extract_rating_counts(ratings_list):
    rating_dict = {}
    total = 0
    for rating in ratings_list:
        rating_dict[rating['name']] = rating['count']
        total += rating['count']
    rating_dict['Total'] = total
    return rating_dict

print("Extracting rating counts...")
rating_features = df_ted['ratings_parsed'].apply(extract_rating_counts)
rating_features_df = pd.DataFrame(rating_features.tolist())
df_ted = pd.concat([df_ted, rating_features_df], axis=1)
print(f"Extracted {len(rating_features_df.columns)} ratings\n")

print("Calculating percentages...")
key_ratings = ['Inspiring', 'Informative', 'Persuasive']
for rating in key_ratings:
    if rating in df_ted.columns:
        df_ted[f"{rating}_Percentage"] = (df_ted[rating] / df_ted['Total']) * 100

print(df_ted[['title', 'Inspiring_Percentage', 'Informative_Percentage', 'Persuasive_Percentage']].head())
print()

WISDOM_PROXY = 'Inspiring_Percentage'
print(f"Using {WISDOM_PROXY} as wisdom proxy")
print(df_ted[WISDOM_PROXY].describe())
print()

q1 = df_ted[WISDOM_PROXY].quantile(0.25)
q3 = df_ted[WISDOM_PROXY].quantile(0.75)
print(f"Q1: {q1:.2f}%")
print(f"Q3: {q3:.2f}%\n")

df_ted['Impact_Group'] = 'Medium'
df_ted.loc[df_ted[WISDOM_PROXY] >= q3, 'Impact_Group'] = 'High'
df_ted.loc[df_ted[WISDOM_PROXY] <= q1, 'Impact_Group'] = 'Low'

group_counts = df_ted['Impact_Group'].value_counts()
print("Groups:")
for group in ['High', 'Medium', 'Low']:
    count = group_counts.get(group, 0)
    print(f"  {group}: {count} talks")
print()

print("="*70)
print("SECTION 3 COMPLETE")
print("="*70 + "\n")

def clean_transcript(text):
    cleaned = re.sub(r'\([^)]*\)', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

print("Cleaning transcripts...")
df_ted['transcript_cleaned'] = df_ted['transcript'].apply(clean_transcript)
print("Done\n")

FIRST_PERSON_SINGULAR = ['i', 'me', 'my', 'mine', 'myself']
FIRST_PERSON_PLURAL = ['we', 'us', 'our', 'ours', 'ourselves']
SECOND_PERSON = ['you', 'your', 'yours', 'yourself', 'yourselves']
COGNITIVE_WORDS = [
    'think', 'thought', 'thinking', 'understand', 'understanding', 'analyze',
    'analysis', 'because', 'reason', 'reasoning', 'consider', 'considered',
    'considering', 'know', 'knew', 'knowledge', 'believe', 'believed', 'belief',
    'realize', 'realized', 'wonder', 'wondered', 'imagine', 'imagined',
    'suppose', 'supposed', 'assume', 'assumed', 'reflect', 'reflected'
]

def extract_features(text):
    doc = nlp(text)
    tokens = [token for token in doc if not token.is_punct and not token.is_space]
    token_texts_lower = [token.text.lower() for token in tokens]
    total_words = len(tokens)
    
    if total_words == 0:
        return {
            'first_person_singular_density': 0,
            'first_person_plural_density': 0,
            'second_person_density': 0,
            'avg_sentence_length': 0,
            'type_token_ratio': 0,
            'cognitive_words_density': 0
        }
    
    fps_count = sum(1 for token in token_texts_lower if token in FIRST_PERSON_SINGULAR)
    fpp_count = sum(1 for token in token_texts_lower if token in FIRST_PERSON_PLURAL)
    sp_count = sum(1 for token in token_texts_lower if token in SECOND_PERSON)
    
    fps_density = fps_count / total_words
    fpp_density = fpp_count / total_words
    sp_density = sp_count / total_words
    
    sentences = list(doc.sents)
    num_sentences = len(sentences)
    avg_sent_length = total_words / num_sentences if num_sentences > 0 else 0
    
    unique_words = set(token_texts_lower)
    ttr = len(unique_words) / total_words
    
    cognitive_count = sum(1 for token in token_texts_lower if token in COGNITIVE_WORDS)
    cognitive_density = cognitive_count / total_words
    
    return {
        'first_person_singular_density': fps_density,
        'first_person_plural_density': fpp_density,
        'second_person_density': sp_density,
        'avg_sentence_length': avg_sent_length,
        'type_token_ratio': ttr,
        'cognitive_words_density': cognitive_density
    }

print("Filtering to high and low impact groups...")
df_analysis = df_ted[df_ted['Impact_Group'].isin(['High', 'Low'])].copy()
print(f"Analyzing {len(df_analysis)} talks\n")

print("Extracting features...")
tqdm.pandas(desc="Processing")
features_extracted = df_analysis['transcript_cleaned'].progress_apply(extract_features)
features_df = pd.DataFrame(features_extracted.tolist(), index=df_analysis.index)
df_analysis = pd.concat([df_analysis, features_df], axis=1)
print("\nFeature extraction done\n")

feature_columns = [
    'first_person_singular_density',
    'first_person_plural_density',
    'second_person_density',
    'avg_sentence_length',
    'type_token_ratio',
    'cognitive_words_density'
]

print(df_analysis[['title', 'Impact_Group'] + feature_columns].head())
print()

df_analysis.to_csv('ted_analysis_processed.csv', index=False)
print("Saved processed data\n")

print("="*70)
print("SECTION 4 COMPLETE")
print("="*70 + "\n")

high_impact = df_analysis[df_analysis['Impact_Group'] == 'High']
low_impact = df_analysis[df_analysis['Impact_Group'] == 'Low']

print(f"Running t-tests on {len(high_impact)} high vs {len(low_impact)} low impact talks\n")

SIGNIFICANCE_LEVEL = 0.05
results = []

for feature in feature_columns:
    high_values = high_impact[feature].dropna()
    low_values = low_impact[feature].dropna()
    
    mean_high = high_values.mean()
    mean_low = low_values.mean()
    
    t_stat, p_value = stats.ttest_ind(high_values, low_values)
    is_significant = p_value < SIGNIFICANCE_LEVEL
    
    pooled_std = np.sqrt(((len(high_values)-1)*high_values.std()**2 + 
                          (len(low_values)-1)*low_values.std()**2) / 
                         (len(high_values) + len(low_values) - 2))
    cohens_d = (mean_high - mean_low) / pooled_std if pooled_std > 0 else 0
    
    results.append({
        'Feature': feature,
        'Mean_High_Impact': mean_high,
        'Mean_Low_Impact': mean_low,
        'Difference': mean_high - mean_low,
        'T_Statistic': t_stat,
        'P_Value': p_value,
        'Significant': is_significant,
        'Cohens_D': cohens_d
    })

results_df = pd.DataFrame(results)
print("Statistical Results:")
print(results_df.to_string(index=False))
print("\n")

significant_features = results_df[results_df['Significant'] == True]
print(f"Found {len(significant_features)} significant differences:\n")

for idx, row in significant_features.iterrows():
    direction = "higher" if row['Difference'] > 0 else "lower"
    effect_size = abs(row['Cohens_D'])
    if effect_size < 0.2:
        effect_desc = "negligible"
    elif effect_size < 0.5:
        effect_desc = "small"
    elif effect_size < 0.8:
        effect_desc = "medium"
    else:
        effect_desc = "large"
    
    print(f"{row['Feature']}")
    print(f"  High: {row['Mean_High_Impact']:.6f}")
    print(f"  Low: {row['Mean_Low_Impact']:.6f}")
    print(f"  Direction: {direction}")
    print(f"  T-stat: {row['T_Statistic']:.4f}, p-value: {row['P_Value']:.6f}")
    print(f"  Effect size: {row['Cohens_D']:.4f} ({effect_desc})")
    print() 

feature_labels = {
    'first_person_singular_density': 'First-Person Singular\nDensity (I, me, my)',
    'first_person_plural_density': 'First-Person Plural\nDensity (we, us, our)',
    'second_person_density': 'Second-Person\nDensity (you, your)',
    'avg_sentence_length': 'Average Sentence\nLength',
    'type_token_ratio': 'Type-Token Ratio\n(Lexical Diversity)',
    'cognitive_words_density': 'Cognitive Words\nDensity'
}

num_features = len(feature_columns)
num_cols = 3
num_rows = (num_features + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5*num_rows))
axes = axes.flatten()

for idx, feature in enumerate(feature_columns):
    ax = axes[idx]
    plot_data = df_analysis[[feature, 'Impact_Group']].copy()
    
    sns.violinplot(data=plot_data, x='Impact_Group', y=feature, 
                   order=['Low', 'High'], palette=['#FF6B6B', '#4ECDC4'],
                   ax=ax)
    
    sns.boxplot(data=plot_data, x='Impact_Group', y=feature,
                order=['Low', 'High'], ax=ax, width=0.3,
                boxprops=dict(alpha=0.7), showcaps=True,
                whiskerprops=dict(linewidth=1.5),
                medianprops=dict(color='black', linewidth=2))
    
    feature_result = results_df[results_df['Feature'] == feature].iloc[0]
    is_sig = feature_result['Significant']
    p_val = feature_result['P_Value']
    
    title = feature_labels.get(feature, feature)
    if is_sig:
        ax.set_title(f"{title}\n(p = {p_val:.4f} *)", fontweight='bold', fontsize=11)
    else:
        ax.set_title(f"{title}\n(p = {p_val:.4f})", fontsize=11)
    
    ax.set_xlabel('Impact Group', fontsize=10)
    ax.set_ylabel('Density / Value', fontsize=10)
    ax.grid(True, alpha=0.3)

for idx in range(num_features, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('linguistic_features_comparison.png', dpi=300, bbox_inches='tight')
print("Saved visualization: linguistic_features_comparison.png\n")

if len(significant_features) > 0:
    sig_feature_list = significant_features['Feature'].tolist()
    num_sig = len(sig_feature_list)
    
    fig2, axes2 = plt.subplots(1, num_sig, figsize=(6*num_sig, 5))
    if num_sig == 1:
        axes2 = [axes2]
    
    for idx, feature in enumerate(sig_feature_list):
        ax = axes2[idx]
        plot_data = df_analysis[[feature, 'Impact_Group']].copy()
        
        sns.violinplot(data=plot_data, x='Impact_Group', y=feature,
                       order=['Low', 'High'], palette=['#FF6B6B', '#4ECDC4'],
                       ax=ax)
        
        sns.boxplot(data=plot_data, x='Impact_Group', y=feature,
                    order=['Low', 'High'], ax=ax, width=0.3,
                    boxprops=dict(alpha=0.7), showcaps=True,
                    whiskerprops=dict(linewidth=1.5),
                    medianprops=dict(color='black', linewidth=2))
        
        feature_result = results_df[results_df['Feature'] == feature].iloc[0]
        p_val = feature_result['P_Value']
        cohens = feature_result['Cohens_D']
        
        title = feature_labels.get(feature, feature)
        ax.set_title(f"{title}\n(p = {p_val:.4f}, d = {cohens:.3f})", 
                     fontweight='bold', fontsize=12)
        ax.set_xlabel('Impact Group', fontsize=11)
        ax.set_ylabel('Density / Value', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('significant_features_only.png', dpi=300, bbox_inches='tight')
    print("Saved focused visualization: significant_features_only.png\n")

results_df.to_csv('statistical_results.csv', index=False)
print("Saved statistical results\n")

print("="*70)
print("Key findings:")
for idx, row in significant_features.iterrows():
    feature = row['Feature']
    direction = "higher" if row['Difference'] > 0 else "lower"
    percentage_diff = abs((row['Difference'] / row['Mean_Low_Impact']) * 100) if row['Mean_Low_Impact'] != 0 else 0
    print(f"- {feature}: {direction} by {percentage_diff:.1f}% in inspiring talks")
print("\n" + "="*70)

print("\nANALYSIS COMPLETE")
print("\nOutputs:")
print("  ted_analysis_processed.csv")
print("  statistical_results.csv")
print("  linguistic_features_comparison.png")
if len(significant_features) > 0:
    print("  significant_features_only.png")
print()