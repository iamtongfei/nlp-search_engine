from sklearn.feature_extraction.text import TfidfVectorizer

# Example Airbnb descriptions
airbnb_descriptions = [
    "Spacious apartment in the heart of the city...",
    "Cozy studio near the beach with ocean view...",
    # Add more Airbnb descriptions here...
]

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the descriptions
tfidf_matrix = tfidf_vectorizer.fit_transform(airbnb_descriptions)

# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get TF-IDF scores for each word
tfidf_scores = tfidf_matrix.sum(axis=0)

# Get top keywords based on TF-IDF scores
top_tfidf_keywords = [feature_names[idx] for idx in tfidf_scores.argsort()[0][-5:]]
print("Top TF-IDF Keywords:", top_tfidf_keywords)
