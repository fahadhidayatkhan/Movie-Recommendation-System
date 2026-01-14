import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and clean
df = pd.read_csv("movies_metadata.csv", low_memory=False)
df = df.dropna(subset=['overview'])
df['overview'] = df['overview'].fillna('')
df['title'] = df['title'].fillna('')

# TF-IDF + cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Recommender
def recommend(title, n=5):
    idx = indices[title]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
    return df['title'].iloc[[i[0] for i in sim_scores]]

# Examples
print("Recommendations for 'The Dark Knight':\n", recommend("The Dark Knight"))
print("\nRecommendations for 'Toy Story':\n", recommend("Toy Story"))