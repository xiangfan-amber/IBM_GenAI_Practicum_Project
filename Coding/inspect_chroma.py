# inspect_chroma.py
import os
from chromadb import PersistentClient

DB_PATH = "./vector_db"

print(f"[inspect] DB path: {os.path.abspath(DB_PATH)}  exists={os.path.exists(DB_PATH)}")
db = PersistentClient(path=DB_PATH)

cols = db.list_collections()
if not cols:
    print("[inspect] No collections found in this DB.")
    raise SystemExit(0)

print("[inspect] Collections found:")
for c in cols:
    try:
        cnt = c.count()
    except Exception:
        cnt = "?"
    print(f"  - name={c.name} | count={cnt}")

# Optional: peek a few docs/metadata from the first collection
c0 = cols[0]
print(f"\n[inspect] Sampling first collection: {c0.name}")
try:
    sample = c0.get(limit=3, include=["documents", "metadatas"])
    print("[inspect] sample ids:      ", sample.get("ids"))
    print("[inspect] sample metadatas:", sample.get("metadatas"))
    docs = sample.get("documents") or []
    for i, d in enumerate(docs, 1):
        preview = (d or "")[:120].replace("\n", " ")
        print(f"[inspect] doc#{i} preview: {preview}")
except Exception as e:
    print("[inspect] sample read error:", e)