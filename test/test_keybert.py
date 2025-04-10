from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample sentence
sentence = "The quick brown fox jumps over the lazy dog"

# Initialize KeyBERT model
kw_model = KeyBERT()

# Extract embeddings for the sentence and candidate words/phrases
doc_embedding = kw_model.model.embed([sentence])[0]
candidates = sentence.split()  # Simple split; you could use n-grams or other tokenization
candidate_embeddings = kw_model.model.embed(candidates)

# Compute cosine similarity between sentence and each candidate
similarities = cosine_similarity([doc_embedding], candidate_embeddings)[0]
for i in range(len(candidates)):
    print(f"Candidate: {candidates[i]}, Similarity: {similarities[i]}")
# Set a similarity threshold instead of top N
threshold = 0.3  # Adjust this value based on your needs
keywords = [(candidates[i], similarities[i]) for i in range(len(candidates)) if similarities[i] > threshold]

# Output all keywords above threshold
print(keywords)