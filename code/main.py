import pandas as pd
import numpy as np
import seaborn as sns


activity_bank = pd.read_csv(r'C:\Users\thars\Downloads\Personalised work for SEND children\data\sample_activity_bank.csv')
child_profile = pd.read_csv(r'C:\Users\thars\Downloads\Personalised work for SEND children\data\child_profiles_150.csv')
print(child_profile.head())
print(activity_bank.head())
print(child_profile.columns)
print(activity_bank.columns)

#Data Cleaning
print(child_profile.isnull().sum())
activity_bank = activity_bank.drop_duplicates()
child_profile = child_profile.drop_duplicates()
child_profile['Preferences'] = child_profile['Preferences'].str.lower().str.strip()
child_profile['Actual_Preferences'] = child_profile['Preferences'].str.split(', ')
child_profile['Diagnosis'] = child_profile['Diagnosis'].astype(str).str.lower().str.strip().str.split(', ')
child_profile['Goals'] = child_profile['Goals'].astype(str).str.lower().str.strip().str.split(', ')
exploded_pref = child_profile.explode('Actual_Preferences')

#EDA
print(exploded_pref['Actual_Preferences'].value_counts())
age_count = child_profile.groupby('Age').size()
print(age_count)
exploded = child_profile.explode('Diagnosis').explode('Goals')
diagnosis_counts = exploded['Diagnosis'].value_counts()


import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
diagnosis_counts.plot(kind='bar', color='blue', edgecolor='black')
plt.title('Number of children by Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Number of children')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


preference_counts = exploded_pref['Actual_Preferences'].value_counts()
top_pref = preference_counts.head(6)

plt.figure(figsize=(8,8))
plt.pie(
    top_pref,
    labels= top_pref.index,
    autopct='%1.1f%%',
    startangle=140,
    wedgeprops={'edgecolor':'black'}
)

plt.title('Preferences among children')
plt.tight_layout()
plt.show()

diag_goal_cross = pd.crosstab(exploded['Diagnosis'],exploded['Goals'])
print(diag_goal_cross)

plt.figure(figsize=(10,6))
sns.heatmap(diag_goal_cross, annot=True, fmt='d', cmap='Blues')
plt.title('Diagnosis vs Goal Frequency')
plt.xlabel('Goals')
plt.ylabel('Diagnosis')
plt.tight_layout()
plt.show()

#Model
#Recommender System
activity_bank['Preferred By'] = activity_bank['Preferred By'].astype(str).str.lower().str.strip().str.split(', ')
exploded_activities = activity_bank.explode('Preferred By')

recommendations = exploded_pref.merge(
    exploded_activities,
    left_on='Actual_Preferences',
    right_on='Preferred By',
    how='inner'
)

grouped_recommendations = recommendations.groupby('ChildID')['Activity Name'].unique().reset_index()
grouped_recommendations.columns = ['ChildID','Recommended Activities']

print(grouped_recommendations.head())

#Clustering Model Optimal Groups for class teaching
first_10 = child_profile.head(10).copy()
print(type(first_10))

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

mlb_pref = MultiLabelBinarizer()
mlb_diag = MultiLabelBinarizer()
mlb_goals = MultiLabelBinarizer()

pref_encoded = pd.DataFrame(mlb_pref.fit_transform(first_10['Actual_Preferences']), columns=mlb_pref.classes_)
diag_encoded = pd.DataFrame(mlb_diag.fit_transform(first_10['Diagnosis']), columns=mlb_diag.classes_)
goals_encoded = pd.DataFrame(mlb_goals.fit_transform(first_10['Goals']), columns=mlb_goals.classes_)
features = pd.concat([
    first_10[['Age']].reset_index(drop=True), 
    pref_encoded, 
    diag_encoded,
    ], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
first_10['Cluster'] = kmeans.fit_predict(X_scaled)

print(first_10[['ChildID','Age','Actual_Preferences','Diagnosis','Cluster']])

sns.scatterplot(data=first_10, x='Age', y='Cluster', hue='Cluster', palette='Set2')
plt.title('Optimized Group Sessions')
plt.xlabel('Age')
plt.ylabel('Cluster')
plt.show()

#Evaluation
print(first_10['Cluster'].value_counts())

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0],reduced[:,1], c=first_10['Cluster'], cmap='Set2', s=100)
plt.title('K-Means Clustering of Children')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import silhouette_score
score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score:{score:.2f}')