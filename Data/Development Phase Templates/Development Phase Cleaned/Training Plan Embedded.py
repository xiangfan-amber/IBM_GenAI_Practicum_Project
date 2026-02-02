import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity #for testing purpose
import numpy as np


with open("Training Plan.json", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for section in data["sections"]:
    if section.get("content"):
        rows.append({
            "document": data["document_title"],
            "section_number": section["number"],
            "title": section["title"],
            "content": section["content"]
        })


    for sub in section.get("subsections", []):
        rows.append({
            "document": data["document_title"],
            "section_number": sub["number"],
            "title": sub["title"],
            "content": sub["content"]
        })

df = pd.DataFrame(rows)



model = SentenceTransformer("all-mpnet-base-v2")
df["embedding"] = df["content"].apply(lambda x: model.encode(x).tolist())
print(df[["section_number", "title", "embedding"]].head(10))
df.to_json("TPembedded.json", orient="records", indent=2, force_ascii=False)


def semantic_search(query, df, top_k=5):
    query_vec = model.encode(query)
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity([x], [query_vec])[0][0])
    return df.sort_values(by="similarity", ascending=False).head(top_k)[["section_number", "title", "similarity"]]

# Example test
results = semantic_search("Who approves the training plan?", df)
print(results)

#Some potential questions to test the similarity
queries = [
    "Who approves the training plan?",
    "What materials are included in the training?",
    "List the key personnel responsible for training coordination.",
    "Where to find the contact of program manager?",
    "Where to find organization name for system development in Training Plan"
]


#print(df[["section_number"]])