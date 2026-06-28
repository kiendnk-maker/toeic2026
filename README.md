# TOEIC Campus Sprint

> A browser-based grammar intervention tool for Vietnamese EFL learners — designed for classroom pilots and active recall research.

**Live:** [toeic.peterswork.shop](https://toeic.peterswork.shop)

---

## What It Does

TOEIC Campus Sprint is a lightweight web app that trains grammar rule retrieval through **spoken production**. Learners are shown a grammatically incorrect TOEIC sentence and must verbalize the correct rule in the form *"Khi thấy … thì dùng …"* ("When you see X, use Y").

The app runs a **randomized two-group design**:

| Group | Condition | Mechanic |
|-------|-----------|----------|
| **exp** | Speak aloud | Mic → Whisper STT → AI rule check |
| **ctrl** | Read silently | Timed dwell → reveal cue + rule |

Both groups see identical grammar content — the only variable is production modality.

---

## Architecture

```
Browser (index.html)
    │
    ├── Static files served by nginx / Cloudflare Tunnel
    │
    └── /api/*  ──►  FastAPI backend (port 8090)
                          ├── POST /api/transcribe    Groq Whisper STT
                          ├── POST /api/check-rule    DeepSeek semantic check
                          ├── POST /api/session/*     SQLite session logging
                          └── GET  /api/health
```

**Stack:**
- Frontend — single-file HTML/CSS/JS (no build step, no framework)
- Backend — FastAPI + Uvicorn (Python 3.10+)
- STT — [Groq Whisper](https://console.groq.com/) (`whisper-large-v3-turbo`)
- Rule check — [DeepSeek](https://platform.deepseek.com/) (`deepseek-v4-pro`)
- Data — SQLite (`sessions.db`)
- Infra — Hetzner VPS + Cloudflare Tunnel + nginx

---

## Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier works)
- A [DeepSeek API key](https://platform.deepseek.com/)

### Install

```bash
git clone https://github.com/kiendnk-maker/toeic2026.git
cd toeic2026
pip install -r backend/requirements.txt
```

### Configure

Create `backend/.env`:

```env
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=sk-...
```

### Run

```bash
# Backend (port 8090)
cd backend
uvicorn main:app --host 127.0.0.1 --port 8090

# Frontend — open index.html directly in browser,
# or serve via any static file server:
python -m http.server 8089
```

Visit `http://localhost:8089`.

---

## Project Structure

```
toeic2026/
├── index.html            # Full frontend (single file)
├── backend/
│   ├── main.py           # FastAPI app — STT, rule check, session DB
│   ├── server.py         # Static file server + /api proxy (single-process mode)
│   ├── start.sh          # VPS start script (uvicorn)
│   └── requirements.txt
└── .gitignore
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/transcribe` | Audio file → transcript (Groq Whisper) |
| `POST` | `/api/check-rule` | Text → PASS/FAIL rule check (DeepSeek) |
| `POST` | `/api/session/start` | Register participant, assign exp/ctrl group |
| `POST` | `/api/session/complete` | Save session results |
| `GET`  | `/api/health` | Service health + key status |

---

## Research Context

This app was built to support a classroom pilot studying **spoken retrieval practice** for grammar acquisition among Vietnamese university EFL learners. The exp/ctrl design isolates the effect of verbal articulation (speak-aloud) versus silent reading on rule retention.

Grammar items target TOEIC Part 5 constructions: present perfect, passive voice, conditionals, relative clauses, and common connectors.

---

## License

MIT
