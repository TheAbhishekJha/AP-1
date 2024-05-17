import pandas as pd, json, numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from utils import get_embeddings

def get_actions_motives_embeddings(df):
    actions = df['action'].tolist()
    motivations = df['motives']

    actions_embeddings = []
    for i in range(len(actions)):
        actions_embeddings.append(get_embeddings(actions[i]))

    motivations_embeddings = []
    for i in range(len(motivations)):
        one_action_motives = []
        if motivations[i]:
            motives = [item['motivation'] for item in json.loads(motivations[i])['Motives']]
            for motive in motives:
                one_action_motives.append(get_embeddings(motive))
        motivations_embeddings.append(one_action_motives)

    all_motives = np.vstack([np.vstack(motives) for motives in motivations_embeddings if motives])

    all_actions = np.vstack(actions_embeddings)

    combined_embeddings = np.vstack((all_actions, all_motives))

    return combined_embeddings

def get_corresponding_texts(df):
    actions = df['action'].tolist()
    text_motives_dict = df['motives']
    motives = []
    for text_dict in text_motives_dict:
        temp = []
        try:
            items = json.loads(text_dict)['Motives']
            for item in items:
                temp.append(item['motivation'])
        except:
            pass
        motives.extend(temp)
    return actions, motives

def call_k_means(df, combined_embeddings, k):
    scaler = StandardScaler()
    combined_embeddings_scaled = scaler.fit_transform(combined_embeddings)

    optimal_k = k 
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(combined_embeddings_scaled)

    # Separate the indices for actions and motives
    num_actions = len(df)  # Total number of actions
    action_indices = np.arange(num_actions)  # Indices of actions in the combined array
    motive_indices = np.arange(num_actions, len(combined_embeddings_scaled))  # Indices of motives in the combined array

    # Find clusters containing both actions and motives
    clusters_with_both = set()
    for cluster_id in range(optimal_k):
        if np.any(clusters[action_indices] == cluster_id) and np.any(clusters[motive_indices] == cluster_id):
            clusters_with_both.add(cluster_id)
    return clusters, clusters_with_both, action_indices, motive_indices

def generate_word_cloud(texts, title):
    """Generate and display a word cloud given a list of texts."""
    text_combined = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400).generate(text_combined)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# For each cluster that has both actions and motives
def call_word_cloud_for_cluster(clusters, cluster_id, actions, motives, action_indices, motive_indices, df):
    # Extract texts for actions and motives within this cluster
    action_texts = [actions[i] for i in action_indices if clusters[i] == cluster_id]
    motive_texts = [motives[j - len(df)] for j in motive_indices if clusters[j] == cluster_id]

    if action_texts:
        generate_word_cloud(action_texts, f'Cluster {cluster_id} Actions Word Cloud')
    
    if motive_texts:
        generate_word_cloud(motive_texts, f'Cluster {cluster_id} Motives Word Cloud')