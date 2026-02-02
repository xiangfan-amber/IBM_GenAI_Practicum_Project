#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 00:11:02 2025

@author: adonischeng
"""

import json
import chromadb

# Load the embedded file you created earlier
with open("TPembedded.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract content and embeddings
documents = [item["content"] for item in data]
embeddings = [item["embedding"] for item in data]


chroma_client = chromadb.PersistentClient(path="./chroma_trainingplan")
collection = chroma_client.get_or_create_collection("training_plan")

ids = [f"section_{i}" for i in range(len(documents))]
collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

# Example query
query = "Who approves the training plan?"

results = collection.query(
    query_texts=[query],
    n_results=3
)

for i in range(len(results["documents"][0])):
    print(f"\n Result {i+1}")
    print(f"Text: {results['documents'][0][i][:200]}...")
    print(f"Similarity Score: {results['distances'][0][i]:.4f}")