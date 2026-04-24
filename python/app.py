from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import pytesseract
from PIL import Image
import faiss
import numpy as np
import pickle
import os
import re

app = Flask(__name__)

# ══════════════════════════════════════════
#  1. LOAD FLAN-T5 (answer generator)
# ══════════════════════════════════════════
model_name = "google/flan-t5-base"
print("⏳ Loading FLAN-T5...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("✅ FLAN-T5 ready!")

# ══════════════════════════════════════════
#  2. LOAD EMBEDDING MODEL (for RAG)
# ══════════════════════════════════════════
print("⏳ Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast on CPU
print("✅ Embedder ready!")

# ══════════════════════════════════════════
#  3. FAISS INDEX & DOCUMENT STORE
# ══════════════════════════════════════════
EMBED_DIM   = 384          # all-MiniLM-L6-v2 output size
INDEX_FILE  = "faiss.index"
CHUNKS_FILE = "chunks.pkl"

# Load existing index if available
if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
    faiss_index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        doc_chunks = pickle.load(f)
    print(f"✅ Loaded existing FAISS index ({len(doc_chunks)} chunks)")
else:
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    doc_chunks  = []          # list of {"text": ..., "source": ...}
    print("✅ New FAISS index created")

# ══════════════════════════════════════════
#  4. HELPERS
# ══════════════════════════════════════════
def chunk_text(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks."""
    words  = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def add_to_index(text, source="upload"):
    """Embed & add text chunks to FAISS."""
    chunks = chunk_text(text)
    for chunk in chunks:
        vec = embedder.encode([chunk]).astype("float32")
        faiss_index.add(vec)
        doc_chunks.append({"text": chunk, "source": source})
    # persist
    faiss.write_index(faiss_index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(doc_chunks, f)
    return len(chunks)

def retrieve(query, top_k=4):
    """Retrieve top-k relevant chunks for a query."""
    if faiss_index.ntotal == 0:
        return []
    vec = embedder.encode([query]).astype("float32")
    distances, indices = faiss_index.search(vec, min(top_k, faiss_index.ntotal))
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1 and dist < 200:          # distance threshold
            results.append(doc_chunks[idx]["text"])
    return results

def generate_answer(question, context_chunks=None):
    """Generate answer using FLAN-T5, with optional RAG context."""
    if context_chunks:
        context = " ".join(context_chunks)
        prompt = f"""You are a helpful AI assistant.
Use the following context to answer the question accurately.

Context: {context}

Question: {question}

Answer:"""
    else:
        prompt = f"Answer the following question clearly and helpfully: {question}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=200,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ══════════════════════════════════════════
#  5. TESSERACT
# ══════════════════════════════════════════
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ══════════════════════════════════════════
#  6. ROUTES
# ══════════════════════════════════════════
@app.route("/")
def home():
    return render_template("index.html")

# ── Chat (RAG-powered) ──
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "Message nahi mila"}), 400

    question = data["message"].strip()
    if not question:
        return jsonify({"response": "Please type something! 😊"})

    # Try RAG first
    chunks = retrieve(question, top_k=4)
    used_rag = len(chunks) > 0

    answer = generate_answer(question, chunks if used_rag else None)

    return jsonify({
        "response": answer,
        "rag_used": used_rag,
        "sources": len(chunks)
    })

# ── Image upload (OCR + index + answer) ──
@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "Image nahi mili"}), 400

    file = request.files["image"]
    img  = Image.open(file)

    extracted = pytesseract.image_to_string(img).strip().replace("\n", " ")
    if not extracted:
        return jsonify({"text": "", "response": "No text found in this image."})

    # Add OCR text to FAISS so future questions can use it
    n = add_to_index(extracted, source="image_ocr")

    # Generate immediate answer/summary
    answer = generate_answer(
        "Summarize and explain this text clearly.",
        [extracted[:800]]
    )

    return jsonify({
        "text": extracted,
        "response": answer,
        "chunks_added": n
    })

# ── Document upload (PDF / TXT) ──
@app.route("/upload_doc", methods=["POST"])
def upload_doc():
    if "document" not in request.files:
        return jsonify({"error": "Document nahi mila"}), 400

    file = request.files["document"]
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(file)
        text   = " ".join(p.extract_text() or "" for p in reader.pages)
    elif filename.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
    else:
        return jsonify({"error": "Only PDF or TXT files supported"}), 400

    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return jsonify({"error": "Document mein koi text nahi mila"}), 400

    n = add_to_index(text, source=filename)
    return jsonify({
        "message": f"✅ Document indexed! {n} chunks added to knowledge base.",
        "chunks": n
    })

# ── Stats ──
@app.route("/stats")
def stats():
    return jsonify({
        "total_chunks": faiss_index.ntotal,
        "documents": len(set(c["source"] for c in doc_chunks))
    })

# ── Clear index ──
@app.route("/clear_index", methods=["POST"])
def clear_index():
    global faiss_index, doc_chunks
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    doc_chunks  = []
    if os.path.exists(INDEX_FILE): os.remove(INDEX_FILE)
    if os.path.exists(CHUNKS_FILE): os.remove(CHUNKS_FILE)
    return jsonify({"message": "✅ Knowledge base cleared!"})

# ══════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True)