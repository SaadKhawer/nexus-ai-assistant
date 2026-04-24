from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import SentenceTransformer
from groq import Groq
import pytesseract
from PIL import Image
import faiss
import numpy as np
import pickle
import os
import re
import json
import requests
import base64
import urllib.parse
import random
from io import BytesIO
from datetime import datetime

app = Flask(__name__)
app.secret_key = "nexusai-secret-key-2024"

# ══════════════════════════════════════════
#  🔑 API KEYS — SIRF FILE MEIN DAALO!
# ══════════════════════════════════════════
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
groq_client       = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL        = "llama-3.1-8b-instant"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
print("✅ Groq + Vision + Stability AI ready!")

# ══════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nexusai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role          = db.Column(db.String(20), default='employee')
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    chats         = db.relationship('Chat', backref='user', lazy=True)

    def set_password(self, pw):   self.password_hash = generate_password_hash(pw)
    def check_password(self, pw): return check_password_hash(self.password_hash, pw)

class Chat(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title      = db.Column(db.String(100), default='New Chat')
    messages   = db.Column(db.Text, default='[]')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def get_messages(self):       return json.loads(self.messages)
    def set_messages(self, msgs): self.messages = json.dumps(msgs)

@login_manager.user_loader
def load_user(user_id): return User.query.get(int(user_id))

# ══════════════════════════════════════════
#  EMBEDDING + FAISS
# ══════════════════════════════════════════
print("⏳ Loading embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedder ready!")

EMBED_DIM   = 384
INDEX_FILE  = "faiss.index"
CHUNKS_FILE = "chunks.pkl"

if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
    faiss_index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        doc_chunks = pickle.load(f)
    print(f"✅ FAISS loaded ({len(doc_chunks)} chunks)")
else:
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    doc_chunks  = []

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

def add_to_index(text, source="upload"):
    chunks = chunk_text(text)
    for chunk in chunks:
        vec = embedder.encode([chunk]).astype("float32")
        faiss_index.add(vec)
        doc_chunks.append({"text": chunk, "source": source})
    faiss.write_index(faiss_index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f: pickle.dump(doc_chunks, f)
    return len(chunks)

def retrieve(query, top_k=2):
    if faiss_index.ntotal == 0: return []
    vec = embedder.encode([query]).astype("float32")
    distances, indices = faiss_index.search(vec, min(top_k, faiss_index.ntotal))
    return [doc_chunks[i]["text"] for i, d in zip(indices[0], distances[0])
            if i != -1 and d < 80]

# ══════════════════════════════════════════
#  GROQ TEXT
# ══════════════════════════════════════════
def ask_groq(question, context_chunks=None, chat_history=None):
    system = "You are NexusAI, a helpful and knowledgeable AI assistant. Answer every question clearly and accurately. Use markdown formatting where helpful."
    if context_chunks:
        context = "\n".join(context_chunks)[:600]
        system += f"\n\nContext from uploaded documents:\n{context}"

    msgs = [{"role": "system", "content": system}]
    if chat_history:
        for m in chat_history[-4:]:
            msgs.append({
                "role": "user" if m['role'] == 'user' else "assistant",
                "content": m['text'][:200]
            })
    msgs.append({"role": "user", "content": question})

    result = groq_client.chat.completions.create(
        model=GROQ_MODEL, messages=msgs,
        max_tokens=512, temperature=0.7, stream=False
    )
    return result.choices[0].message.content.strip()

# ══════════════════════════════════════════
#  🔍 GROQ VISION
# ══════════════════════════════════════════
def analyze_image_with_groq(image_b64, user_question="", mime_type="image/jpeg"):
    try:
        question = user_question if user_question else "Analyze this image in detail. Describe what you see, extract any text, identify any errors or issues, and provide helpful insights."

        # Resize + convert to JPEG properly
        img_data = base64.b64decode(image_b64)
        img      = Image.open(BytesIO(img_data))
        img      = img.convert('RGB')        # PNG/RGBA → RGB
        img.thumbnail((1024, 1024))
        buffer   = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        mime_type = "image/jpeg"

        response = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }],
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Vision error: {str(e)}"

# ══════════════════════════════════════════
#  🎨 IMAGE GENERATION — Stability AI
# ══════════════════════════════════════════
def generate_image(prompt):
    try:
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={
                "Authorization": f"Bearer {STABILITY_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            json={
                "text_prompts": [{"text": prompt, "weight": 1}],
                "cfg_scale": 7,
                "height": 1024,
                "width": 1024,
                "samples": 1,
                "steps": 20
            },
            timeout=60
        )
        if response.status_code == 200:
            img_b64 = response.json()["artifacts"][0]["base64"]
            return {"success": True, "image": img_b64}
        else:
            return {"success": False, "error": f"Error {response.status_code}: {response.text[:150]}"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout — try again"}
    except Exception as e:
        return {"success": False, "error": str(e)}

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ══════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════
@app.route('/login', methods=['GET','POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('home'))
    if request.method == 'POST':
        data = request.json
        user = User.query.filter_by(username=data.get('username')).first()
        if user and user.check_password(data.get('password','')):
            login_user(user, remember=True)
            return jsonify({"success": True, "role": user.role})
        return jsonify({"success": False, "error": "Invalid credentials"}), 401
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('home'))
    if request.method == 'POST':
        data = request.json
        if User.query.filter_by(username=data.get('username')).first():
            return jsonify({"success": False, "error": "Username already exists"}), 400
        if User.query.filter_by(email=data.get('email')).first():
            return jsonify({"success": False, "error": "Email already exists"}), 400
        u = User(username=data['username'], email=data['email'])
        u.set_password(data['password'])
        db.session.add(u); db.session.commit()
        login_user(u)
        return jsonify({"success": True})
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user(); return redirect(url_for('login'))

# ══════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════
@app.route('/')
@login_required
def home():
    return render_template('index.html', user=current_user)

@app.route('/admin')
@login_required
def admin():
    if current_user.role != 'admin': return redirect(url_for('home'))
    return render_template('admin.html', user=current_user)

# ── Chats ──
@app.route('/chats', methods=['GET'])
@login_required
def get_chats():
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.desc()).all()
    return jsonify([{"id":c.id,"title":c.title,
                     "preview":(c.get_messages() or [{}])[0].get('text','')[:40]} for c in chats])

@app.route('/chats', methods=['POST'])
@login_required
def create_chat():
    c = Chat(user_id=current_user.id)
    db.session.add(c); db.session.commit()
    return jsonify({"id":c.id,"title":c.title})

@app.route('/chats/<int:cid>', methods=['GET'])
@login_required
def get_chat(cid):
    c = Chat.query.filter_by(id=cid, user_id=current_user.id).first_or_404()
    return jsonify({"id":c.id,"title":c.title,"messages":c.get_messages()})

@app.route('/chats/<int:cid>', methods=['DELETE'])
@login_required
def delete_chat(cid):
    c = Chat.query.filter_by(id=cid, user_id=current_user.id).first_or_404()
    db.session.delete(c); db.session.commit()
    return jsonify({"success":True})

# ── ASK ──
@app.route('/ask_stream', methods=['POST'])
@login_required
def ask_stream():
    data     = request.json
    question = data.get('message','').strip()
    chat_id  = data.get('chat_id')
    if not question:
        return jsonify({"error":"Empty message"}), 400

    chat    = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first() if chat_id else None
    history = chat.get_messages() if chat else []
    chunks  = retrieve(question)
    rag     = len(chunks) > 0

    try:
        answer = ask_groq(question, chunks if rag else None, history)
    except Exception as e:
        return jsonify({"response": f"⚠️ Error: {str(e)}", "rag_used": False})

    if chat:
        m = chat.get_messages()
        m.append({"role":"user","text":question})
        m.append({"role":"bot","text":answer,"rag":rag})
        if len(m) == 2: chat.title = question[:50]
        chat.set_messages(m)
        db.session.commit()

    return jsonify({"response": answer, "rag_used": rag})

# ── CHAT IMAGE ANALYSIS (Vision) — FIXED ──
@app.route('/chat_image', methods=['POST'])
@login_required
def chat_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400

    file     = request.files['image']
    question = request.form.get('question', '')
    chat_id  = request.form.get('chat_id', '')

    # ✅ Read + convert + resize properly
    img_bytes = file.read()
    img       = Image.open(BytesIO(img_bytes))
    img       = img.convert('RGB')       # PNG/RGBA → RGB
    img.thumbnail((1024, 1024))
    buffer    = BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    img_b64   = base64.b64encode(buffer.getvalue()).decode('utf-8')
    mime_type = 'image/jpeg'

    # Analyze with Groq Vision
    answer = analyze_image_with_groq(img_b64, question, mime_type)

    # Save to chat
    if chat_id:
        try:
            chat = Chat.query.filter_by(id=int(chat_id), user_id=current_user.id).first()
            if chat:
                m = chat.get_messages()
                user_text = f"📷 Shared an image{': ' + question if question else ''}"
                m.append({"role":"user","text": user_text, "has_image": True})
                m.append({"role":"bot","text": answer})
                if len(m) == 2: chat.title = "Image Analysis"
                chat.set_messages(m)
                db.session.commit()
        except: pass

    return jsonify({
        "response": answer,
        "image_b64": img_b64,
        "mime_type": mime_type
    })

# ── OCR Upload ──
@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'image' not in request.files: return jsonify({"error":"No image"}), 400
    img  = Image.open(request.files['image'])
    text = pytesseract.image_to_string(img).strip().replace("\n"," ")
    if not text: return jsonify({"text":"","response":"No text found."})
    add_to_index(text, source="image_ocr")
    answer = ask_groq("Summarize and explain this text clearly.", [text[:600]])
    return jsonify({"text":text,"response":answer})

# ── Document upload ──
@app.route('/upload_doc', methods=['POST'])
@login_required
def upload_doc():
    if 'document' not in request.files: return jsonify({"error":"No doc"}), 400
    file = request.files['document']
    if file.filename.lower().endswith('.pdf'):
        import pypdf
        text = " ".join(p.extract_text() or "" for p in pypdf.PdfReader(file).pages)
    elif file.filename.lower().endswith('.txt'):
        text = file.read().decode('utf-8', errors='ignore')
    else:
        return jsonify({"error":"Only PDF or TXT"}), 400
    text = re.sub(r'\s+', ' ', text).strip()
    if not text: return jsonify({"error":"No text found"}), 400
    n = add_to_index(text, source=file.filename)
    return jsonify({"message":f"✅ {n} chunks added!", "chunks":n})

# ── Stats ──
@app.route('/stats')
@login_required
def stats():
    return jsonify({
        "total_chunks": faiss_index.ntotal,
        "documents":    len(set(c["source"] for c in doc_chunks))
    })

# ── Clear KB ──
@app.route('/clear_index', methods=['POST'])
@login_required
def clear_index():
    global faiss_index, doc_chunks
    if current_user.role != 'admin': return jsonify({"error":"Admin only"}), 403
    faiss_index = faiss.IndexFlatL2(EMBED_DIM); doc_chunks = []
    for f in [INDEX_FILE, CHUNKS_FILE]:
        if os.path.exists(f): os.remove(f)
    return jsonify({"message":"✅ Cleared!"})

# ── Image Generation ──
@app.route('/generate_image', methods=['POST'])
@login_required
def gen_image():
    data   = request.json
    prompt = data.get('prompt','').strip()
    if not prompt:
        return jsonify({"success": False, "error": "Prompt khali hai"}), 400
    enhanced = f"{prompt}, high quality, detailed, professional"
    result   = generate_image(enhanced)
    return jsonify(result)

# ── Admin Stats ──
@app.route('/admin/stats')
@login_required
def admin_stats():
    if current_user.role != 'admin': return jsonify({"error":"Forbidden"}), 403
    from collections import Counter
    from datetime import timedelta
    users      = User.query.all()
    total_chat = Chat.query.count()
    total_msg  = sum(len(c.get_messages()) for c in Chat.query.all())
    by_user    = [{"username":u.username,"email":u.email,
                   "chats":Chat.query.filter_by(user_id=u.id).count(),
                   "role":u.role} for u in users]
    today     = datetime.utcnow().date()
    days      = [(today - timedelta(days=i)).isoformat() for i in range(6,-1,-1)]
    day_count = Counter(c.created_at.date().isoformat() for c in Chat.query.all())
    daily     = [{"date":d,"count":day_count.get(d,0)} for d in days]
    return jsonify({"total_users":len(users),"total_chats":total_chat,
                    "total_messages":total_msg,"by_user":by_user,
                    "daily":daily,"kb_chunks":faiss_index.ntotal})

@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'admin': return jsonify({"error":"Forbidden"}), 403
    return jsonify([{"id":u.id,"username":u.username,"email":u.email,"role":u.role}
                    for u in User.query.all()])

@app.route('/admin/users/<int:uid>/role', methods=['POST'])
@login_required
def change_role(uid):
    if current_user.role != 'admin': return jsonify({"error":"Forbidden"}), 403
    u = User.query.get_or_404(uid)
    u.role = request.json.get('role','employee')
    db.session.commit()
    return jsonify({"success":True})

# ══════════════════════════════════════════
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            a = User(username='admin', email='admin@nexusai.com', role='admin')
            a.set_password('admin123')
            db.session.add(a); db.session.commit()
            print("✅ Admin: admin / admin123")
    app.run(
        host     = "0.0.0.0",
        port     = int(os.environ.get("PORT", 5000)),
        debug    = False,
        threaded = True
    )