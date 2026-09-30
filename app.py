from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import chromadb
import os
import shutil
import logging
from fastapi.staticfiles import StaticFiles

from sage_engine import ingest, retrieve, ask_llm
from prompt_builder import build_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="sage_docs")


@app.get("/", response_class=HTMLResponse)
async def home():
        return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SAGE — Document Q&A</title>
        <link rel="icon" type="image/png" href="/static/sage_logo.png">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }

            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #05070d;
                background-image:
                    radial-gradient(circle at 15% 10%, rgba(120, 180, 255, 0.12), transparent 45%),
                    radial-gradient(circle at 85% 15%, rgba(180, 120, 255, 0.10), transparent 45%),
                    radial-gradient(circle at 50% 90%, rgba(100, 220, 150, 0.06), transparent 50%);
                background-attachment: fixed;
                color: #e8eef7;
                height: 100vh;
                overflow: hidden;
                display: flex;
            }

            /* ---------- SIDEBAR ---------- */
            .sidebar {
                width: 280px;
                background: rgba(15, 20, 32, 0.6);
                backdrop-filter: blur(30px) saturate(160%);
                -webkit-backdrop-filter: blur(30px) saturate(160%);
                border-right: 1px solid rgba(180, 210, 255, 0.08);
                padding: 1.5rem 1rem;
                display: flex;
                flex-direction: column;
                overflow-y: auto;
                transition: transform 0.3s ease;
                flex-shrink: 0;
            }

            .logo {
                display: flex;
                align-items: center;
                gap: 0.7rem;
                padding: 0.5rem 0.5rem 1.5rem;
                border-bottom: 1px solid rgba(180, 210, 255, 0.08);
                margin-bottom: 1.5rem;
            }
            .logo img { height: 36px; }
            .logo h1 {
                font-size: 1.4rem;
                font-weight: 600;
                background: linear-gradient(135deg, #ffffff 0%, #a8c8ff 60%, #c9b4ff 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                letter-spacing: -0.5px;
            }

            .upload-zone {
                background: rgba(20, 28, 45, 0.5);
                border: 1.5px dashed rgba(180, 210, 255, 0.2);
                border-radius: 12px;
                padding: 1.2rem 1rem;
                text-align: center;
                cursor: pointer;
                transition: all 0.2s ease;
                margin-bottom: 1.5rem;
            }
            .upload-zone:hover, .upload-zone.dragover {
                border-color: rgba(88, 166, 255, 0.5);
                background: rgba(88, 166, 255, 0.08);
            }
            .upload-zone .icon {
                font-size: 1.4rem;
                margin-bottom: 0.4rem;
                opacity: 0.7;
            }
            .upload-zone p {
                font-size: 0.82rem;
                color: #9db4d4;
                line-height: 1.4;
            }
            .upload-zone .browse {
                color: #a8ccff;
                text-decoration: underline;
            }

            .docs-label {
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #6e7f9a;
                margin-bottom: 0.8rem;
                padding-left: 0.5rem;
            }

            .doc-list {
                display: flex;
                flex-direction: column;
                gap: 0.3rem;
                flex: 1;
            }

            .doc-item {
                padding: 0.6rem 0.8rem;
                border-radius: 10px;
                font-size: 0.85rem;
                color: #c9d6e8;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                transition: background 0.15s ease;
                word-break: break-word;
                line-height: 1.3;
            }
            .doc-item:hover { background: rgba(180, 210, 255, 0.06); }
            .doc-item.active {
                background: rgba(88, 166, 255, 0.12);
                color: #a8ccff;
            }
            .doc-item .bullet { color: #6e7f9a; font-size: 0.9rem; }

            .empty-docs {
                padding: 0.6rem 0.8rem;
                font-size: 0.8rem;
                color: #6e7f9a;
                font-style: italic;
            }

            /* ---------- MAIN AREA ---------- */
            .main {
                flex: 1;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

            .topbar {
                padding: 1rem 1.5rem;
                border-bottom: 1px solid rgba(180, 210, 255, 0.08);
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: rgba(10, 15, 25, 0.4);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
            }
            .topbar-left {
                display: flex;
                align-items: center;
                gap: 0.8rem;
                font-size: 0.9rem;
                color: #9db4d4;
            }
            .topbar select {
                background: rgba(20, 28, 45, 0.5);
                border: 1px solid rgba(180, 210, 255, 0.12);
                color: #e8eef7;
                padding: 0.5rem 0.8rem;
                border-radius: 8px;
                font-size: 0.85rem;
                font-family: inherit;
                cursor: pointer;
            }
            .topbar select:focus { outline: none; border-color: rgba(88, 166, 255, 0.4); }

            .new-chat-btn {
                background: rgba(88, 166, 255, 0.1);
                border: 1px solid rgba(88, 166, 255, 0.2);
                color: #a8ccff;
                padding: 0.5rem 0.9rem;
                border-radius: 8px;
                font-size: 0.85rem;
                cursor: pointer;
                font-family: inherit;
                transition: background 0.2s ease;
            }
            .new-chat-btn:hover { background: rgba(88, 166, 255, 0.2); }

            .chat-area {
                flex: 1;
                overflow-y: auto;
                padding: 1.5rem;
                display: flex;
                flex-direction: column;
            }

            .empty-state {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    text-align: center;
}

.empty-inner {
    max-width: 500px;
    width: 100%;
}

.empty-icon {
    font-size: 2.5rem;
    color: #a8ccff;
    background: rgba(88, 166, 255, 0.08);
    border: 1px solid rgba(88, 166, 255, 0.2);
    width: 64px;
    height: 64px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1.5rem;
}

.empty-state h2 {
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
    color: #e8eef7;
}

.empty-state p {
    color: #9db4d4;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

.big-drop {
    padding: 2rem 1.5rem;
    border: 1.5px dashed rgba(180, 210, 255, 0.2);
    border-radius: 16px;
    cursor: pointer;
    transition: all 0.2s ease;
    background: rgba(20, 28, 45, 0.35);
}

.big-drop:hover, .big-drop.dragover {
    border-color: rgba(88, 166, 255, 0.5);
    background: rgba(88, 166, 255, 0.08);
}

.big-drop p { color: #c9d6e8; font-size: 0.95rem; margin-bottom: 0.3rem; }
.big-drop .small { color: #6e7f9a; font-size: 0.78rem; margin-bottom: 0; }

            /* ---------- MESSAGES ---------- */
            .message {
                max-width: 85%;
                padding: 0.9rem 1.2rem;
                border-radius: 14px;
                margin-bottom: 0.8rem;
                line-height: 1.6;
                font-size: 0.95rem;
                white-space: pre-wrap;
                word-wrap: break-word;
            }

            .message.user {
                align-self: flex-end;
                background: linear-gradient(135deg, #1db954 0%, #25d366 100%);
                color: #000;
                font-weight: 500;
                border-bottom-right-radius: 4px;
            }

            .message.bot {
                align-self: flex-start;
                background: rgba(20, 28, 45, 0.6);
                border: 1px solid rgba(180, 210, 255, 0.1);
                color: #dce6f2;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-bottom-left-radius: 4px;
            }

            .message.bot.error {
                background: rgba(255, 138, 138, 0.08);
                border-color: rgba(255, 138, 138, 0.25);
                color: #ff8a8a;
            }

            .message.loading {
                align-self: flex-start;
                color: #9db4d4;
                font-style: italic;
                font-size: 0.9rem;
                padding-left: 0.4rem;
            }

            /* ---------- INPUT ---------- */
            .input-area {
                padding: 1rem 1.5rem 1.5rem;
                border-top: 1px solid rgba(180, 210, 255, 0.08);
                background: rgba(10, 15, 25, 0.5);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
            }
            .input-wrap {
                display: flex;
                background: rgba(20, 28, 45, 0.6);
                border: 1px solid rgba(180, 210, 255, 0.12);
                border-radius: 14px;
                padding: 0.4rem 0.4rem 0.4rem 1rem;
                align-items: center;
                transition: border-color 0.2s ease;
            }
            .input-wrap:focus-within { border-color: rgba(88, 166, 255, 0.4); }

            .input-wrap input {
                flex: 1;
                background: transparent;
                border: none;
                color: #e8eef7;
                font-size: 0.95rem;
                font-family: inherit;
                padding: 0.6rem 0;
                outline: none;
            }
            .input-wrap input::placeholder { color: #6e7f9a; }

            .send-btn {
                background: #1db954;
                color: #000;
                border: none;
                width: 38px;
                height: 38px;
                border-radius: 10px;
                cursor: pointer;
                font-size: 1rem;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background 0.2s ease, transform 0.1s ease;
            }
            .send-btn:hover { background: #25d366; }
            .send-btn:active { transform: scale(0.96); }
            .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }

            .status-line {
                font-size: 0.78rem;
                color: #6e7f9a;
                margin-top: 0.5rem;
                padding-left: 0.2rem;
                min-height: 1rem;
            }
            .status-line.success { color: #6ee79b; }
            .status-line.error { color: #ff8a8a; }

            /* ---------- MOBILE ---------- */
            .menu-toggle {
                display: none;
                background: transparent;
                border: none;
                color: #a8ccff;
                font-size: 1.3rem;
                cursor: pointer;
                padding: 0.3rem;
            }

            .overlay {
                display: none;
                position: fixed;
                inset: 0;
                background: rgba(0, 0, 0, 0.5);
                z-index: 9;
            }

            @media (max-width: 768px) {
                .sidebar {
                    position: fixed;
                    top: 0;
                    left: 0;
                    height: 100vh;
                    transform: translateX(-100%);
                    z-index: 10;
                }
                .sidebar.open { transform: translateX(0); }
                .menu-toggle { display: block; }
                .overlay.show { display: block; }
                .empty-state h2 { font-size: 1.3rem; }
            }
        </style>
    </head>
    <body>

        <div class="overlay" id="overlay"></div>

        <aside class="sidebar" id="sidebar">
            <div class="logo">
                <img src="/static/sage_logo.png" alt="SAGE">
                <h1>SAGE</h1>
            </div>

            <div class="upload-zone" id="sidebarUpload">
                <div class="icon">⬆</div>
                <p>Drop a PDF or <span class="browse">click to browse</span></p>
            </div>
            <input type="file" id="fileInput" accept="application/pdf" style="display: none;">

            <div class="docs-label">Documents</div>
            <div class="doc-list" id="docList">
                <div class="empty-docs">No documents yet.</div>
            </div>
        </aside>

        <main class="main">
            <div class="topbar">
                <div class="topbar-left">
                    <button class="menu-toggle" id="menuToggle">☰</button>
                    <select id="sourceSelect">
                        <option value="">All documents</option>
                    </select>
                </div>
                <button class="new-chat-btn" id="newChatBtn">+ New chat</button>
            </div>

            <div class="chat-area" id="chatArea">
    <div class="empty-state" id="emptyState">
        <div class="empty-inner">
            <div class="empty-icon">⬆</div>
            <h2>Start by adding a document</h2>
            <p>Upload a PDF, then ask questions about it.</p>
            <div class="big-drop" id="bigDrop">
                <p><strong>Drop files or click to upload</strong></p>
                <p class="small">PDF files supported</p>
            </div>
        </div>
    </div>
</div>

            <div class="input-area">
                <form id="askForm">
                    <div class="input-wrap">
                        <input type="text" id="questionInput" placeholder="Ask about your documents..." autocomplete="off" required>
                        <button type="submit" class="send-btn" id="sendBtn">↑</button>
                    </div>
                </form>
                <div class="status-line" id="statusLine"></div>
            </div>
        </main>

        <script>
        const fileInput = document.getElementById('fileInput');
        const sidebarUpload = document.getElementById('sidebarUpload');
        const bigDrop = document.getElementById('bigDrop');
        const docList = document.getElementById('docList');
        const sourceSelect = document.getElementById('sourceSelect');
        const chatArea = document.getElementById('chatArea');
        const emptyState = document.getElementById('emptyState');
        const askForm = document.getElementById('askForm');
        const questionInput = document.getElementById('questionInput');
        const sendBtn = document.getElementById('sendBtn');
        const statusLine = document.getElementById('statusLine');
        const sidebar = document.getElementById('sidebar');
        const menuToggle = document.getElementById('menuToggle');
        const overlay = document.getElementById('overlay');
        const newChatBtn = document.getElementById('newChatBtn');

        // ---------- UPLOAD ----------
        function triggerFilePicker() { fileInput.click(); }
        sidebarUpload.onclick = triggerFilePicker;
        bigDrop.onclick = triggerFilePicker;

        fileInput.onchange = async () => {
            if (fileInput.files.length > 0) {
                await uploadFile(fileInput.files[0]);
                fileInput.value = '';
            }
        };

        // Drag & drop
        [sidebarUpload, bigDrop].forEach(zone => {
            zone.addEventListener('dragover', e => {
                e.preventDefault();
                zone.classList.add('dragover');
            });
            zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
            zone.addEventListener('drop', async e => {
                e.preventDefault();
                zone.classList.remove('dragover');
                if (e.dataTransfer.files.length > 0) {
                    await uploadFile(e.dataTransfer.files[0]);
                }
            });
        });

        async function uploadFile(file) {
            if (!file.name.toLowerCase().endsWith('.pdf')) {
                setStatus('Only PDF files are supported.', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);
            setStatus(`Uploading ${file.name}...`, '');

            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.message) {
                    setStatus(data.message, 'success');
                    await loadDocuments();
                } else {
                    setStatus(data.error || 'Upload failed.', 'error');
                }
            } catch {
                setStatus('Upload failed. Check your connection.', 'error');
            }
        }

        function setStatus(text, type) {
            statusLine.innerText = text;
            statusLine.className = 'status-line' + (type ? ' ' + type : '');
            if (type === 'success') {
                setTimeout(() => { statusLine.innerText = ''; statusLine.className = 'status-line'; }, 4000);
            }
        }

        // ---------- DOCUMENTS ----------
        async function loadDocuments() {
            try {
                const res = await fetch('/documents');
                const data = await res.json();
                const docs = data.documents || [];

                // Populate sidebar
                docList.innerHTML = '';
                if (docs.length === 0) {
                    docList.innerHTML = '<div class="empty-docs">No documents yet.</div>';
                } else {
                    docs.forEach(doc => {
                        const item = document.createElement('div');
                        item.className = 'doc-item';
                        item.innerHTML = `<span class="bullet">●</span> ${doc}`;
                        item.onclick = () => {
                            sourceSelect.value = doc;
                            markActiveDoc(doc);
                            closeSidebarOnMobile();
                        };
                        docList.appendChild(item);
                    });
                }

                // Populate dropdown
                sourceSelect.innerHTML = '<option value="">All documents</option>';
                docs.forEach(doc => {
                    const opt = document.createElement('option');
                    opt.value = doc;
                    opt.innerText = doc;
                    sourceSelect.appendChild(opt);
                });

                // Update empty state visibility
                if (docs.length === 0) {
                    emptyState.style.display = 'flex';
                } else {
                    emptyState.style.display = 'none';
                }
            } catch (err) {
                console.error(err);
            }
        }

        function markActiveDoc(doc) {
            document.querySelectorAll('.doc-item').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.doc-item').forEach(el => {
                if (el.innerText.includes(doc)) el.classList.add('active');
            });
        }

        // ---------- ASK ----------
        askForm.onsubmit = async (e) => {
            e.preventDefault();
            const question = questionInput.value.trim();
            if (!question) return;

            const source = sourceSelect.value;
            questionInput.value = '';
            sendBtn.disabled = true;

            addMessage(question, 'user');
            const loadingId = addMessage('Thinking...', 'loading');

            try {
                const formData = new FormData();
                formData.append('question', question);
                if (source) formData.append('source', source);

                const res = await fetch('/ask', { method: 'POST', body: formData });
                const data = await res.json();

                removeMessage(loadingId);

                if (data.error) {
                    addMessage(data.error, 'bot error');
                } else {
                    addMessage(data.answer, 'bot');
                }
            } catch {
                removeMessage(loadingId);
                addMessage('Something went wrong. Please try again.', 'bot error');
            } finally {
                sendBtn.disabled = false;
                questionInput.focus();
            }
        };

        function addMessage(text, type) {
            emptyState.style.display = 'none';
            const el = document.createElement('div');
            el.className = 'message ' + type;
            el.innerText = text;
            chatArea.appendChild(el);
            chatArea.scrollTop = chatArea.scrollHeight;
            if (type === 'loading') el.id = 'loadingMsg';
            return 'loadingMsg';
        }

        function removeMessage(id) {
            const el = document.getElementById(id);
            if (el) el.remove();
        }

        // ---------- NEW CHAT ----------
        newChatBtn.onclick = () => {
            // Keep the sidebar and dropdown. Just wipe the chat messages.
            document.querySelectorAll('.message').forEach(el => el.remove());
            emptyState.style.display = sourceSelect.value === '' ? 'flex' : 'none';
            setStatus('', '');
        };

        // ---------- MOBILE SIDEBAR ----------
        menuToggle.onclick = () => {
            sidebar.classList.add('open');
            overlay.classList.add('show');
        };
        overlay.onclick = closeSidebarOnMobile;

        function closeSidebarOnMobile() {
            sidebar.classList.remove('open');
            overlay.classList.remove('show');
        }

        // ---------- INIT ----------
        loadDocuments();
        </script>
    </body>
    </html>
    """)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingest(file_path, collection)
    return {"message": f"Uploaded and indexed: {file.filename}"}


@app.get("/sources")
async def list_sources():
    results = collection.get(include=["metadatas"])
    sources = set()
    for meta in results["metadatas"]:
        if meta and "source" in meta:
            sources.add(meta["source"])
    return {"sources": sorted(sources)}

@app.get("/documents")
async def list_documents():
    results = collection.get(include=["metadatas"])
    sources = set()
    for meta in results["metadatas"]:
        if meta and "source" in meta:
            sources.add(meta["source"])
    return {"documents": sorted(sources)}

@app.post("/ask")
async def ask(question: str = Form(...), source: str = Form(None)):
    try:
        chunks = retrieve(question, collection, source=source)
        prompt = build_prompt(question, chunks)
        answer = ask_llm(prompt)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Ask failed: {e}")
        return {"error": "The AI service is temporarily unavailable. Please try again in a minute."}