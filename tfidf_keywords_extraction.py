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

## MOCK DATA
airbnb_description = "Large, private, modern, spacious apartment in luxury high-rise elevator building located near the the East River overlooking Brooklyn & Manhattan. Floor-to-ceiling windows throughout - with an incredible amount of natural light during the day & dazzling city views at night. Centrally located in the heart of Williamsburg near endless amazing restaurants, grocery stores, bars, cafes, and more! One stop to Manhattan on the L or JMZ trains."
human_input = "modern, high-rise, scenic views of Brooklyn & Manhattan, ample natural light, Williamsburg location with varied amenities, quick access to Manhattan via L or JMZ trains."

## STEP 02: EXTRACT THE KEYWRODS FROM DECRIPTION
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# Tokenize and perform part-of-speech tagging
tokens = word_tokenize(airbnb_description)
pos_tags = pos_tag(tokens)

# Extract noun phrases using POS tagging
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
tree = cp.parse(pos_tags)

noun_phrases = [' '.join(leaf[0] for leaf in subtree.leaves()) for subtree in tree.subtrees() if subtree.label() == 'NP']
print("Noun Phrases:", noun_phrases)

## STEP 03: COMPARE THE KEYWRODS WITH USER INPUTS
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

# Load a pre-trained Word2Vec model
filepath = 'GoogleNews-vectors-negative300.bin.gz'
#word2vec_model = KeyedVectors.load_word2vec_format(filepath, binary=True)
word2vec_model = KeyedVectors.load_word2vec_format(filepath, binary=True)

# Example sentence and list of words
sentence = "Your input sentence here."
word_list = noun_phrases

# Tokenize the sentence and word list
tokens_sentence = human_input
tokens_word_list = [word.lower() for word in word_list]

# Compute word embeddings for sentence and word list
embeddings_sentence = [word2vec_model[word] for word in tokens_sentence if word in word2vec_model.vocab]
embeddings_word_list = [word2vec_model[word] for word in tokens_word_list if word in word2vec_model.vocab]

# Compute cosine similarity between sentence and each word in the list
similarity_scores = [cosine_similarity([embeddings_sentence], [embedding_word])[0][0] for embedding_word in embeddings_word_list]

# Sort words based on similarity scores and select top 5
top_similar_words = [word_list[index] for index in sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:5]]

print("Top 5 most similar words:", top_similar_words)
