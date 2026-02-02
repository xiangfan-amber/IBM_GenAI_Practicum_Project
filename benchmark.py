import os, json, glob, re, csv, time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

DATA_DIR = "data"
OUT_FILE = "result.csv"
MAX_CONTEXT_CHARS = 12000
MODELS = ["gpt-5-nano", "gpt-4o-mini", "gpt-4o"]
QUESTIONS = [
    "List key deliverables in the Development Phase and briefly describe each.",
    "What are the responsibilities of the Project Manager during Development?",
    "What are the exit criteria for the Development Phase?"
]
SYSTEM_PROMPT = (
    "You are an EPLC assistant. Answer strictly based on the provided context. "
    "If information is missing, reply exactly: 'Not specified in the provided context.' "
    "Return clear and structured text."
)

def _read_text_any_encoding(fp):
    for enc in ("utf-8","utf-16","utf-16-le","utf-16-be","latin-1"):
        try:
            with open(fp,"r",encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""

def extract_text_from_json(obj, buf):
    KEYS = {"title","section","section_title","content","summary","description",
            "responsibilities","deliverables","exit","criteria","activities",
            "objective","scope","review","artifact","output","role"}
    NOISE = re.compile(r"(DOCPROPERTY|MERGEFORMAT)", re.I)
    def _push(t):
        t = t.strip()
        if len(t) > 20 and not NOISE.search(t): buf.append(t)
    if isinstance(obj, dict):
        for k,v in obj.items():
            if isinstance(v,str) and any(h in k.lower() for h in KEYS): _push(v)
        for k,v in obj.items():
            if isinstance(v,str) and not any(h in k.lower() for h in KEYS): _push(v)
        for v in obj.values():
            if isinstance(v,(dict,list)): extract_text_from_json(v,buf)
    elif isinstance(obj,list):
        for it in obj: extract_text_from_json(it,buf)

def load_context(data_dir, max_chars):
    files = glob.glob(os.path.join(data_dir,"*.json"))
    print(f"[DEBUG] JSON files found: {len(files)}")
    texts=[]
    for fp in files:
        raw=_read_text_any_encoding(fp)
        try:
            j=json.loads(raw); buf=[]; extract_text_from_json(j,buf)
            text="\n".join(buf).strip()
        except Exception:
            text=raw.strip()
        if text:
            block=f"\n### FILE: {os.path.basename(fp)}\n{text}\n"
            texts.append(block)
            print(f"[DEBUG] {os.path.basename(fp)} -> added {len(block)} chars")
    full="\n".join(texts).strip()
    if len(full)>max_chars: full=full[:max_chars]
    print(f"[DEBUG] Context length: {len(full)}")
    print("[DEBUG] Context preview:\n",full[:800],"\n--- END PREVIEW ---")
    return full

def ask(client, model, context, question):
    t0=time.time()
    resp=client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
    )
    latency=round(time.time()-t0,3)
    msg=resp.choices[0].message.content or ""
    usage=resp.usage
    return latency, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, msg

def main():
    print("[DEBUG] CWD:", os.getcwd())
    if not API_KEY: raise ValueError("❌ OPENAI_API_KEY not found in .env")
    client=OpenAI(api_key=API_KEY)
    context=load_context(DATA_DIR,MAX_CONTEXT_CHARS)
    assert len(context)>200, "Context too short. Check your JSON files."

    with open(OUT_FILE,"w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["model","question","latency_s","input_tokens","output_tokens","total_tokens","answer_preview"])

        for q in QUESTIONS:
            for m in MODELS:
                try:
                    latency,ti,to,tt,text=ask(client,m,context,q)
                    writer.writerow([m,q,latency,ti,to,tt,text[:200].replace("\n"," ")])
                    print(f"[OK] {m} | {q[:40]}... | {latency}s | tokens={tt}")
                except Exception as e:
                    print(f"[ERROR] {m} | {q[:40]}... | {e}")
                    writer.writerow([m,q,0,0,0,0,f"ERROR: {e}"])

    print(f"\n✅ Saved: {OUT_FILE}")

if __name__=="__main__":
    main()