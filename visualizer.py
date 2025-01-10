import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load the data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure for feature distributions
plt.figure(figsize=(15, 10))
for i, feature in enumerate(data.feature_names[:5], 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=df, x=feature, hue='target', bins=30, alpha=0.6)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=False)
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.show()

# Box plots for key features
plt.figure(figsize=(15, 10))
features_to_plot = data.feature_names[:5]
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=df, x='target', y=feature)
    plt.title(f'Box Plot of {feature}')
    plt.xlabel('Target (0: Malignant, 1: Benign)')
plt.tight_layout()
plt.show()

# Scatter plot matrix for select features
selected_features = data.feature_names[:3]
selected_df = df[list(selected_features) + ['target']]
sns.pairplot(selected_df, hue='target', diag_kind='hist')
plt.show()

# Feature importance using mean values
plt.figure(figsize=(12, 6))
mean_benign = df[df['target'] == 1][data.feature_names].mean()
mean_malignant = df[df['target'] == 0][data.feature_names].mean()
feature_importance = pd.DataFrame({
    'Benign': mean_benign,
    'Malignant': mean_malignant
}, index=data.feature_names)
feature_importance.plot(kind='bar')
plt.title('Mean Feature Values for Benign vs Malignant Tumors')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Class distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='target')
plt.title('Distribution of Target Classes')
plt.xlabel('Target (0: Malignant, 1: Benign)')
plt.ylabel('Count')
plt.show()
