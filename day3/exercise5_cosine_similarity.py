import math

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    return dot_product / (magnitude1 * magnitude2)


query = [0.9, 0.1, 0.8]

documents = {
    "AI strategy document": [0.8, 0.2, 0.9],
    "Cricket coaching guide": [0.1, 0.9, 0.2],
    "Cloud architecture paper": [0.7, 0.3, 0.8]
}

for title, vector in documents.items():
    score = cosine_similarity(query, vector)
    print(title, "=>", round(score, 3))
