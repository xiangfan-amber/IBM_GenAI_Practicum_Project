# Import packages
import os, sys
from typing import List, Tuple
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Runtime and performance settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Environment setup
load_dotenv()

DB_PATH       = os.getenv("CHROMA_PATH", "./vector_db")
COLL_NAME = os.getenv("CHROMA_COLLECTION", "Implementation_Phase")
TOP_K         = int(os.getenv("TOP_K", "6"))
CHAT_MODEL    = os.getenv("CHAT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in .env")

# Initialize embedding model
print("[init] Loading SBERT model ...")
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# Connect vector database
print(f"[init] Opening Chroma database at {DB_PATH}...")
db = PersistentClient(path=DB_PATH)
print("[debug] DB abs path:", os.path.abspath(DB_PATH))
print("[debug] collections now:", [c.name for c in db.list_collections()])

coll = db.get_or_create_collection(name=COLL_NAME)

# Database probe utility
def probe_index(c):
    try:
        peek = c.get(limit=1, include=["documents", "metadatas", "embeddings", "ids"])
        print("[probe] id:", peek["ids"][0])
        meta0 = (peek.get("metadatas") or [None])[0] or {}
        doc0 = (peek.get("documents") or [None])[0]
        if not doc0:
            for k in ("text", "content", "page_content", "chunk", "raw"):
                if k in meta0 and meta0[k]:
                    doc0 = meta0[k]
                    break
        print("[probe] doc preview:", (doc0 or "")[:200].replace("\n", " "))
        emb_dim = len((peek.get("embeddings") or [[]])[0])
        print("[probe] embedding dim:", emb_dim)
        return emb_dim
    except Exception as e:
        print("[probe] error:", e)
        return None

# Embedding dimension validation
emb_dim = probe_index(coll)
try:
    _probe_vec = sbert.encode(["test"], normalize_embeddings=True)[0]
    model_dim = len(_probe_vec)
    if emb_dim and model_dim != emb_dim:
        print(f"Embedding dim mismatch: collection={emb_dim}, model={model_dim}")
        sys.exit(1)
    else:
        print(f"[check] ok: collection={emb_dim}, model={model_dim}")
except Exception as e:
    print("[check] error while validating embedding dim:", e)
    sys.exit(1)

# Initialize OpenAI client
oa = OpenAI(api_key=OPENAI_API_KEY)

# Core embedding and retrieval utilities
def embed(texts: List[str]) -> List[List[float]]:
    return sbert.encode(texts, normalize_embeddings=True).tolist()

def retrieve(query: str, k: int = TOP_K) -> Tuple[list, list, list]:
    qv = embed([query])
    res = coll.query(
        query_embeddings=qv,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, dists

def pretty_sim(dist: float) -> float:
    try:
        return 1.0 - float(dist)
    except Exception:
        return float("nan")

def retrieve_exact(substring: str, k: int = TOP_K) -> Tuple[list, list, list]:
    res = coll.get(
        where_document={"$contains": substring},
        include=["documents", "metadatas"],
        limit=k,
    )
    ids   = res.get("ids", [])
    docs  = res.get("documents", [])
    dists = [0.0] * len(docs)
    return ids, docs, dists

# Prompt construction and LLM interaction
SYSTEM_PROMPT = (
    "You are an EPLC assistant. Answer only using the information in the CONTEXT. "
    "If the answer can be inferred from the context, explain it briefly. "
    "If the context provides no relevant information, reply exactly: Not specified in the provided context."
)

def make_prompt(question: str, docs: List[str]) -> str:
    context = "\n\n---\n\n".join(docs)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n"

def ask_openai(prompt: str) -> str:
    try:
        resp = oa.responses.create(
            model=CHAT_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        return f"[openai error] {e}"

# Interactive main loop
def main():
    try:
        cnt = coll.count()
        print(f"[startup] Loaded collection '{COLL_NAME}' with {cnt} records.")
    except Exception as e:
        print("[startup] Collection error:", e)
        sys.exit(1)

    print(f"[ready] Using GPT model: {CHAT_MODEL} | top_k={TOP_K}")
    print("Ask any EPLC question. Type 'exit' to quit.")

    while True:
        try:
            q = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if not q or q.lower() in {"exit", "quit"}:
            print("bye.")
            break

        ids, docs, dists = retrieve_exact(q, TOP_K)
        if not docs:
            ids, docs, dists = retrieve(q, TOP_K)
        if not docs:
            print("A> Not specified in the provided context.")
            continue

        prompt = make_prompt(q, docs)
        answer = ask_openai(prompt)
        print("\nA>", answer if answer else "Not specified in the provided context.")
        print("   citations:", ids)

        # Debug
        print(f"\n[DEBUG] ids={ids}")
        for i, (d, dist) in enumerate(zip(docs, dists), start=1):
            prev = (d or "")[:200].replace("\n", " ")
            print(f"[DEBUG ctx#{i}] dist={dist:.3f} | sim≈{pretty_sim(dist):.3f} | {prev}")


if __name__ == "__main__":
    main()