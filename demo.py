from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

quotes = [
    "The onboarding was really confusing, I didn't know where to start",
    "I gave up during setup because there were too many steps",
    "I couldn't figure out how to invite my teammates",
    "The pricing page wasn't clear, I didn't know what I'd be charged",
    "I had no idea what the free plan included",
    "The search feature is incredibly fast and accurate",
    "I love how quickly I can find past research",
    "Finding old interviews is so much easier now",
]

def find_patterns(quotes, threshold=0.35):
    print("Generating embeddings...\n")
    embeddings = model.encode(quotes)
    similarity_matrix = cosine_similarity(embeddings)
    
    clusters = []
    used = set()
    
    for i in range(len(quotes)):
        if i in used:
            continue
        cluster = [quotes[i]]
        used.add(i)
        for j in range(i + 1, len(quotes)):
            if j not in used and similarity_matrix[i][j] > threshold:
                cluster.append(quotes[j])
                used.add(j)
        clusters.append(cluster)
    
    return clusters

if __name__ == "__main__":
    clusters = find_patterns(quotes)
    
    print("=== INTERVIEW INSIGHT EXTRACTOR ===")
    print(f"Analyzed {len(quotes)} quotes from user interviews\n")
    print("PATTERNS FOUND:")
    print("-" * 40)
    
    for cluster in clusters:
        if len(cluster) > 1:
            print(f"\nPattern ({len(cluster)} similar responses):")
            for q in cluster:
                print(f"  - {q}")
    
    print("\n" + "=" * 40)
    print("Same cosine similarity logic as REMIND, applied to interview data.")
