"""
standup_backend/main.py
FastAPI backend for TOEIC Campus Sprint at toeic.peterswork.shop

Endpoints:
  POST /api/transcribe   — Groq Whisper STT (audio → text)
  POST /api/check-rule   — DeepSeek semantic check (did user name the grammar rule?)
  GET  /api/health       — health check
"""

import os, re, json, time, uuid, random, sqlite3, subprocess, pathlib, unicodedata
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import requests
from dotenv import load_dotenv

# ── Load env (try toeic-pilot .env.production first, then Hermes .env) ──
load_dotenv(pathlib.Path("/home/peter/toeic-pilot/backend/.env.production"))
load_dotenv(pathlib.Path.home() / ".hermes" / ".env", override=False)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GROQ_STT_URL  = "https://api.groq.com/openai/v1/audio/transcriptions"
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"
CHECK_RULE_MODEL = os.getenv("CHECK_RULE_MODEL", "deepseek-v4-pro")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

app = FastAPI(title="TOEIC Campus Sprint Backend", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Simple rate-limit (100 req/min per IP) ─────────────────────────────
_rate: dict[str, list[float]] = {}
def _check_rate(ip: str, limit: int = 100, window: int = 60) -> bool:
    now = time.time()
    bucket = _rate.setdefault(ip, [])
    while bucket and bucket[0] < now - window:
        bucket.pop(0)
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True


# ──────────────────────────────────────────────────────────────────────
# GET /api/health
# ──────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "groq_key": bool(GROQ_API_KEY),
        "deepseek_key": bool(DEEPSEEK_API_KEY),
        "check_rule_model": CHECK_RULE_MODEL,
        "debrief_model": DEEPSEEK_MODEL,
    }


# ──────────────────────────────────────────────────────────────────────
# POST /api/transcribe
# Body: audio file (webm/mp4/wav), lang (en|vi|tw)
# Returns: { text: "..." }
# ──────────────────────────────────────────────────────────────────────
@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    lang:  str = Form("en"),
    gp:    str = Form(""),       # grammar point id, e.g. "past_perfect_bytime"
    point: str = Form(""),       # scaffold cue word, e.g. "by the time"
    accept_kw: str = Form("[]", alias="accept"), # JSON array of accept keywords from frontend
):
    if not GROQ_API_KEY:
        return JSONResponse({"error": "GROQ_API_KEY not configured"}, status_code=503)

    audio_bytes = await audio.read()
    if len(audio_bytes) < 1000:
        return JSONResponse({"error": "Audio too short"}, status_code=400)

    # Detect MIME
    header = audio_bytes[:12]
    if header.startswith(b'\x1aE\xdf\xa3'):
        fname, mime = "audio.webm", "audio/webm"
    elif b'ftyp' in header:
        fname, mime = "audio.mp4", "audio/mp4"
    elif header.startswith(b'RIFF'):
        fname, mime = "audio.wav", "audio/wav"
    else:
        fname, mime = "audio.webm", "audio/webm"

    # Do NOT force "vi" — Vietnamese EFL learners code-switch heavily
    # (e.g. "khi thấy since thì dùng present perfect").
    # Forcing lang="vi" causes Whisper to phonetically transcribe English
    # grammar terms ("since"→"xin", "present perfect"→"quá khứ hoàn thành").
    # large-v3 handles code-switching better without a language hint.
    whisper_lang = {"tw": "zh"}.get(lang, None)

    # Build per-item prompt: grammar point + cue words. Biases Whisper toward
    # the expected vocabulary WITHOUT revealing the answer, so transcript stays
    # faithful to the user's spoken audio (research validity preserved).
    # Fallback to a generic TOEIC vocabulary prompt when caller doesn't supply
    # per-item cues.
    if gp or point:
        gp_label = {
            "present_perfect_since":      "present perfect (since, for)",
            "past_perfect_bytime":        "past perfect (by the time, had)",
            "plural_numberof":            "a number of (plural verb)",
            "singular_each":              "each / every (singular verb)",
            "prep_collocation":           "preposition collocation (responsible for)",
            "gerund_after_prep":          "gerund after preposition (V-ing)",
            "comparative_than":           "comparative (than, -er, more)",
            "first_conditional":          "first conditional (if + present, will)",
            "subject_verb_neither":       "neither of (singular verb)",
            "passive_present_perfect":    "passive present perfect (has/have been + V3)",
        }.get(gp, gp.replace("_", " "))
        cue = point.strip() or gp_label
        # Whisper prompt is transcript-style context, not instructions. Keep it
        # compact because Groq/Whisper only consumes a short prompt window.
        domain_prompt = (
            f"khi thấy {cue} thì dùng {gp_label}. "
            "since, for, present perfect, past perfect, by the time, had, "
            "have been, has been, each, every, neither, than, will, V-ing, passive."
        )
    else:
        domain_prompt = (
            "khi thấy since thì dùng present perfect. khi thấy by the time thì dùng past perfect. "
            "a number of, each, every, neither, than, will, V-ing, passive."
        )

    try:
        kw_list = json.loads(accept_kw) if accept_kw else []
        if kw_list:
            # Put item-specific rare tokens near the end of the prompt.
            domain_prompt = (domain_prompt + " " + ", ".join(kw_list[:6]) + ".")[-900:]
    except Exception:
        pass

    try:
        resp = requests.post(
            GROQ_STT_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (fname, audio_bytes, mime)},
            data={
                # whisper-large-v3 (full) is more accurate than -turbo on Vietnamese
                # accented English; latency cost is ~1-2s, acceptable for rule-check.
                "model": "whisper-large-v3",
                "response_format": "json",
                **({"language": whisper_lang} if whisper_lang else {}),
                # temperature 0.2 allows model to pick better hypothesis on retry
                # (greedy 0.0 freezes same wrong decoding across retries).
                "temperature": "0.2",
                "prompt": domain_prompt,
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json().get("text", "").strip()
        return {"text": text}
    except requests.HTTPError as e:
        return JSONResponse({"error": f"Groq STT error: {e.response.status_code}"}, status_code=502)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ──────────────────────────────────────────────────────────────────────
# POST /api/check-rule
# Body: text (what user said), tense (grammar label), accept (keywords[])
# Returns: { passed: bool, score: 0-1, feedback: "..." }
# ──────────────────────────────────────────────────────────────────────

FILLER = re.compile(r'\b(um|uh|so|like|well|ah|er|hmm|okay|yeah|yes|no)\b', re.I)

# Groq Whisper sometimes writes Vietnamese-with-English terms phonetically
# ("Ki thay sin..." for "Khi thấy since..."). Normalize only common rule words
# so the verifier catches what the learner intended without injecting answers.
_ASR_FIXES = [
    (r"\bki\s+thay\b", "khi thay"),
    (r"\bxin\b", "since"),   # since → xin (Whisper phonetic)
    (r"\bsin\b", "since"),   # since → sin
    (r"\byung\b", "dung"),
    (r"\bhing\s+tai\b", "hien tai"),
    (r"\bhoang\s+thanh\b", "hoan thanh"),
    (r"\bqua\s+khu\s+hoang\s+thanh\b", "qua khu hoan thanh"),
]

def _strip_accents(t: str) -> str:
    t = unicodedata.normalize("NFD", t)
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn").replace("đ", "d").replace("Đ", "D")

def _repair_vi_asr(t: str) -> str:
    fixed = _strip_accents(t.lower())
    for pat, repl in _ASR_FIXES:
        fixed = re.sub(pat, repl, fixed)
    return re.sub(r'\s+', ' ', fixed).strip()

def _normalize(t: str) -> str:
    return re.sub(r'\s+', ' ', FILLER.sub('', _repair_vi_asr(t))).strip()

def _keyword_check(text: str, accept: list[str]) -> bool:
    t = _normalize(text)
    return any(_normalize(k) in t for k in accept)

def _contradicts_rule(text: str, gp: str, tense: str) -> str:
    """Reject transcripts that name the cue but the wrong target tense."""
    t = _normalize(text)
    target = _normalize(f"{gp} {tense}")
    says_past_perfect = bool(re.search(r'\b(past perfect|qua khu hoan thanh|had\s*\+?\s*v3|had\b)\b', t))
    says_present_perfect = bool(re.search(r'\b(present perfect|hien tai hoan thanh|have\s*\+?\s*v3|has\s*\+?\s*v3|have been|has been)\b', t))

    if "present_perfect_since" in gp or "present perfect" in target:
        if says_past_perfect and not says_present_perfect:
            return "Chưa đúng — quy tắc bạn nói chưa khớp với câu này."

    if "past_perfect_bytime" in gp or "past perfect" in target:
        if says_present_perfect and not says_past_perfect:
            return "Chưa đúng — quy tắc bạn nói chưa khớp với câu này."

    return ""

def _is_prompt_echo(text: str, gp: str) -> bool:
    """Reject transcripts that simply echo the Whisper prompt back. Whisper
    large-v3 sometimes hallucinates exactly the domain prompt when the audio
    is silent or too short, which would otherwise pass the keyword check.
    Heuristic: very short, perfect-form Vietnamese template using the cue
    word plus expected rule word(s), with no discourse fillers or natural
    noise. Real learner speech is longer or messier than this.
    """
    t = _normalize(text)
    # Real learners either speak longer than 60 chars, or include fillers /
    # restarts / partial words. Hallucinated echoes are clean and short.
    if len(t) > 60:
        return False
    if "present_perfect_since" in gp:
        # Echo detection uses the RAW (pre-normalize) text so that phonetic
        # ASR variants (xin/sin/sinh → since) are not mistaken for Whisper
        # hallucination. Whisper hallucination of the prompt would produce
        # `since` verbatim, while a real learner who got ASR-distorted would
        # produce `xin`/`sin`/`sinh` in the raw transcript.
        raw = (text or "").lower()
        has_raw_real_cue = re.search(r"\b(since|for)\b", raw)
        has_raw_phonetic_cue = re.search(r"\b(xin|sin|sinh)\b", raw)
        has_rule = re.search(r"\b(present perfect|hien tai hoan thanh|have|has|v3|hoan thanh)\b", t)
        template = re.search(r"khi\s*thay\b.*?\bthi\s*dung\b", t)
        # Strip template words so fillers like `um` survive the accent strip.
        raw2 = re.sub(r"\bthì\b|\bthi\b|\bkhi\s*thấy\b|\bkhi\s*thay\b|\bdùng\b|\bdung\b", " ", raw)
        fillers = re.search(r"\b(cai|thôi|thoi|nhe|nhé|um|uh|ah|oi|được|duoc|à|a)\b", raw2)
        return bool(has_raw_real_cue and has_rule and template and not fillers) \
            and not has_raw_phonetic_cue
    if "past_perfect_bytime" in gp:
        has_cue = re.search(r"\bby\s*the\s*time\b", t)
        has_rule = re.search(r"\b(past perfect|qua khu hoan thanh|had|v3|hoan thanh)\b", t)
        template = re.search(r"khi\s*thay\b.*?\bthi\s*dung\b", t)
        # Strip the cue phrase and trailing "thì" before filler check, because
        # `by the time` and `thì` are part of the grammar template itself,
        # not discourse fillers. Use raw text so discourse fillers like `um`,
        # `cai`, `thôi` survive the strip-accents pass.
        raw = (text or "").lower()
        raw = re.sub(r"by\s+the\s+time", " ", raw)
        raw = re.sub(r"\bthì\b|\bthi\b", " ", raw)
        fillers = re.search(r"\b(cai|thôi|thoi|nhe|nhé|um|uh|ah|oi|được|duoc|à|a)\b", raw)
        return bool(has_cue and has_rule and template and not fillers)
    return False

def _missing_rule_meaning(text: str, gp: str, tense: str) -> str:
    """Reject cue-only or overly generic answers before keyword matching."""
    t = _normalize(text)
    target = _normalize(f"{gp} {tense}")

    if "passive_present_perfect" in gp or "passive present perfect" in target:
        has_passive = re.search(r'\b(passive|bi dong)\b', t)
        has_present_perfect = re.search(r'\b(present perfect|hien tai hoan thanh|has been|have been|been\b)\b', t)
        if has_passive and not has_present_perfect:
            return "Chưa đủ — hãy nói rõ quy tắc ngữ pháp."
        return ""

    if "present_perfect_since" in gp or "present perfect" in target:
        has_cue = re.search(r'\b(since|for|sinh|sin|xin)\b', t)
        has_rule = re.search(r'\b(present perfect|hien tai hoan thanh|have\b|has\b|v3|hoan thanh)\b', t)
        if not has_rule:
            return "Chưa đủ — hãy nói rõ quy tắc ngữ pháp."
        if not has_cue:
            return "Chưa đủ — hãy nói cả dấu hiệu và quy tắc ngữ pháp."

    if "past_perfect_bytime" in gp or "past perfect" in target:
        has_cue = re.search(r'\bby\s*the\s*time\b', t)
        has_rule = re.search(r'\b(past perfect|qua khu hoan thanh|had\b|v3|hoan thanh)\b', t)
        if not has_rule:
            return "Chưa đủ — hãy nói rõ quy tắc ngữ pháp."
        if not has_cue:
            return "Chưa đủ — hãy nói cả dấu hiệu và quy tắc ngữ pháp."

    return ""

@app.post("/api/check-rule")
async def check_rule(
    text:    str = Form(...),
    tense:   str = Form(""),
    accept:  str = Form("[]"),   # JSON array of accepted keywords
    lang:    str = Form("en"),
    gp:      str = Form(""),
    item_id: str = Form(""),
):
    if not DEEPSEEK_API_KEY:
        return JSONResponse({"error": "DEEPSEEK_API_KEY not configured"}, status_code=503)

    try:
        accept_list: list[str] = json.loads(accept)
    except Exception:
        accept_list = []

    cleaned = _normalize(text)
    if len(cleaned) < 12 or re.fullmatch(r'[a-d]\.?', cleaned):
        return {"passed": False, "score": 0.0,
                "feedback": "Name the grammar rule, not just the answer letter.", "method": "reject"}

    contradiction = _contradicts_rule(text, gp, tense)
    if contradiction:
        return {"passed": False, "score": 0.0,
                "feedback": contradiction, "method": "contradiction"}

    if _is_prompt_echo(text, gp):
        return {"passed": False, "score": 0.0,
                "feedback": "Chưa nghe rõ — hãy nói to và rõ hơn vào mic.", "method": "hallucination"}

    missing_rule = _missing_rule_meaning(text, gp, tense)
    if missing_rule:
        return {"passed": False, "score": 0.0,
                "feedback": missing_rule, "method": "cue_only"}

    # Comparative items need the actual comparative form, not just the phrase
    # "so sánh hơn". This prevents ASR-garbled answers like "dạng den" from
    # being inferred as correct by the LLM.
    if gp == "comparative_than":
        if item_id == "P19" and not re.search(r'\b(more|efficient|than)\b', cleaned):
            return {"passed": False, "score": 0.0,
                    "feedback": "Hãy nói rõ dạng so sánh hơn: more + tính từ dài + than.", "method": "strict"}
        if item_id in ("P20", "P21") and not re.search(r'\b(er|higher|lower|than)\b', cleaned):
            return {"passed": False, "score": 0.0,
                    "feedback": "Hãy nói rõ dạng so sánh hơn: thêm -er và dùng than.", "method": "strict"}

    # Fast path: keyword match (no LLM call, free)
    if _keyword_check(text, accept_list):
        return {"passed": True, "score": 1.0, "feedback": "✓ You named the rule.", "method": "keyword"}

    # LLM-based rule verification (DeepSeek v4 pro semantic fallback)
    lang_labels = {"vi": "Vietnamese", "tw": "Traditional Chinese"}
    ui_lang = lang_labels.get(lang, "English")

    system = (
        "You verify whether a Vietnamese EFL learner correctly named a TOEIC grammar rule. "
        "The transcript may be Vietnamese-English code-switching and may contain ASR distortion "
        "(e.g., since→xin/sin, dùng→yung, hiện tại hoàn thành may be unaccented). "
        "Pass ONLY if the learner's meaning matches the target grammar rule. "
        "A cue word alone is NOT enough. The learner must also name the correct tense, structure, "
        "or rule meaning. "
        "Reject contradictory tense/structure claims even if a keyword matches. "
        "Examples: if target is Present Perfect with since/for, reject 'since → past perfect'. "
        "If target is Past Perfect with by the time, reject 'by the time → present perfect'. "
        "Do not pass answer-letter-only responses. "
        "Feedback must not reveal the correct tense, structure, cue word, or answer. "
        "For failures, use generic Vietnamese feedback only, such as: Chưa đủ — hãy nói rõ quy tắc ngữ pháp. "
        "Reply ONLY with valid JSON: "
        "{\"passed\": true/false, \"score\": 0.0-1.0, \"feedback\": \"one short non-informative sentence\"}"
    )
    user = (
        f"Target rule: {tense}\n"
        f"Grammar point id: {gp}\n"
        f"Item id: {item_id}\n"
        f"Expected keywords / acceptable phrasings: {', '.join(accept_list)}\n"
        f"Learner transcript ({ui_lang}): \"{text}\"\n\n"
        "Decision rule:\n"
        "- PASS if the learner names the correct grammar meaning, even with minor ASR distortion.\n"
        "- FAIL if they only say the cue word without the correct rule.\n"
        "- FAIL if they name the opposite/wrong tense or structure.\n"
        "- FAIL if they only say an answer letter or option.\n"
        "- Feedback must not reveal the correct answer, tense, structure, or cue word.\n"
        "- If FAIL, use generic Vietnamese feedback only.\n"
        "Return JSON only."
    )

    try:
        resp = requests.post(
            DEEPSEEK_CHAT_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": CHECK_RULE_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "temperature": 0.0,
                # v4-pro may spend tokens on reasoning before final content; keep
                # enough budget so the JSON answer is not truncated/empty.
                "max_tokens": 1024,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        raw = (msg.get("content") or "").strip()
        if not raw:
            raise ValueError("DeepSeek returned empty content")
        result = json.loads(raw)
        return {
            "passed":   bool(result.get("passed", False)),
            "score":    float(result.get("score", 0.0)),
            "feedback": result.get("feedback", ""),
            "method":   "llm",
        }
    except Exception as e:
        # Fallback: keyword miss → fail
        return {"passed": False, "score": 0.0,
                "feedback": "Chưa đủ — hãy nói rõ quy tắc ngữ pháp.", "method": "fallback"}


# ──────────────────────────────────────────────────────────────────────
# SESSION DB — lightweight SQLite for debrief storage
# ──────────────────────────────────────────────────────────────────────
DB_PATH = pathlib.Path("/home/peter/standup-backend/sessions.db")


def _add_column_if_missing(conn, table: str, col: str, decl: str):
    """Idempotent ALTER TABLE ... ADD COLUMN (SQLite has no IF NOT EXISTS for columns)."""
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if col not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    # ── sessions (one play-through = pre+practice+post) ──
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            created_at  REAL,
            group_type  TEXT,
            pre_score   INTEGER,
            post_score  INTEGER,
            practice_score INTEGER,
            attempts    TEXT,   -- JSON array (legacy, kept)
            debrief     TEXT    -- Hermes output, cached
        )
    """)
    # Migrations — new columns added without dropping existing data.
    _add_column_if_missing(conn, "sessions", "participant_id",  "TEXT")
    _add_column_if_missing(conn, "sessions", "phase",           "TEXT DEFAULT 'post'")
    _add_column_if_missing(conn, "sessions", "paas_germane",    "INTEGER")
    _add_column_if_missing(conn, "sessions", "paas_extraneous", "INTEGER")

    # ── participants (stable identity across pre/post/delayed) ──
    conn.execute("""
        CREATE TABLE IF NOT EXISTS participants (
            participant_id TEXT PRIMARY KEY,
            created_at     REAL,
            group_type     TEXT,    -- 'exp' / 'ctrl' / NULL (not yet assigned)
            pre_total      INTEGER,
            assigned_at    REAL,
            is_admin       INTEGER DEFAULT 0,  -- 1 = P9999 / debug; excluded from analysis
            source         TEXT DEFAULT 'enroll'  -- 'enroll' (random), 'admin' (manual)
        )
    """)
    # Migrations for older DBs that pre-date is_admin / source columns.
    _add_column_if_missing(conn, "participants", "is_admin", "INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "participants", "source",    "TEXT DEFAULT 'enroll'")
    _add_column_if_missing(conn, "participants", "age", "INTEGER")
    _add_column_if_missing(conn, "participants", "gender", "TEXT")
    _add_column_if_missing(conn, "participants", "years_english", "TEXT")
    _add_column_if_missing(conn, "participants", "toeic_exp", "TEXT")
    conn.commit(); conn.close()

_init_db()


# ──────────────────────────────────────────────────────────────────────
# POST /api/enroll
# Body (JSON, optional): { participant_id: "P001".."P999" }
#   - If provided and matches P{NNN} with NNN in 001..999, server checks
#     uniqueness and rejects duplicates. Used by users typing their own
#     assigned number on the intro screen.
#   - If omitted (legacy behavior), server auto-generates uuid8.
#   - Special case: "P9999" (4-digit) is NOT accepted here — see
#     /api/enroll/admin for the debug/admin flow.
# Returns: { participant_id }
# ──────────────────────────────────────────────────────────────────────
import re
_PARTICIPANT_RE = re.compile(r"^P([0-9]{3})$")

@app.post("/api/enroll")
async def enroll(request: Request):
    try:
        data = await request.json() or {}
    except Exception:
        data = {}

    requested = (data.get("participant_id") or "").strip().upper()
    if requested and not _PARTICIPANT_RE.match(requested):
        return JSONResponse(
            {"error": "participant_id must be P001..P999 (3 digits) or omitted"},
            status_code=400,
        )

    conn = sqlite3.connect(DB_PATH)

    if requested:
        # Reject duplicates — each participant gets a unique 3-digit number.
        existing = conn.execute(
            "SELECT 1 FROM participants WHERE participant_id=?", (requested,)
        ).fetchone()
        if existing:
            conn.close()
            return JSONResponse(
                {"error": f"participant_id '{requested}' already exists — pick another"},
                status_code=409,
            )
        pid = requested
    else:
        # Legacy auto-generated 8-char id (kept for backward compat).
        pid = str(uuid.uuid4())[:8]

    conn.execute(
        "INSERT INTO participants (participant_id, created_at, group_type, pre_total, assigned_at, is_admin, source) "
        "VALUES (?,?,?,?,?,?,?)",
        (pid, time.time(), None, None, None, 0, "enroll"),
    )
    conn.commit(); conn.close()
    return {"participant_id": pid}


# ──────────────────────────────────────────────────────────────────────
# POST /api/participants/demographics
# Body: { participant_id, age, gender, years_english, toeic_exp }
# Saves required baseline survey before pre-test starts.
# ──────────────────────────────────────────────────────────────────────
@app.post("/api/participants/demographics")
async def save_demographics(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    pid = (data.get("participant_id") or "").strip().upper()
    gender = (data.get("gender") or "").strip()
    years_english = (data.get("years_english") or "").strip()
    toeic_exp = (data.get("toeic_exp") or "").strip()

    try:
        age = int(data.get("age"))
    except Exception:
        return JSONResponse({"error": "age must be an integer"}, status_code=400)

    valid_gender = {"Nam", "Nữ", "Khác"}
    valid_years = {"<5", "5–8", "9–12", ">12"}
    valid_toeic = {"Chưa", "Có, dưới 500", "Có, 500–700", "Có, trên 700"}
    if not pid:
        return JSONResponse({"error": "participant_id required"}, status_code=400)
    if age < 10 or age > 80:
        return JSONResponse({"error": "age must be between 10 and 80"}, status_code=400)
    if gender not in valid_gender:
        return JSONResponse({"error": "invalid gender"}, status_code=400)
    if years_english not in valid_years:
        return JSONResponse({"error": "invalid years_english"}, status_code=400)
    if toeic_exp not in valid_toeic:
        return JSONResponse({"error": "invalid toeic_exp"}, status_code=400)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "UPDATE participants SET age=?, gender=?, years_english=?, toeic_exp=? WHERE participant_id=?",
        (age, gender, years_english, toeic_exp, pid),
    )
    conn.commit(); conn.close()
    if cur.rowcount == 0:
        return JSONResponse({"error": "participant_not_found"}, status_code=404)
    return {"ok": True}


# ──────────────────────────────────────────────────────────────────────
# POST /api/enroll/admin
# Debug / QA flow. Caller supplies BOTH the chosen group (exp/ctrl) and
# an id like "P9999" so it's visually distinct from real participants.
# Marked is_admin=1 so analysis can filter out these rows.
# Body: { participant_id: "P9999", group: "exp"|"ctrl" }
# Returns: { participant_id, group, is_admin: true }
# ──────────────────────────────────────────────────────────────────────
@app.post("/api/enroll/admin")
async def enroll_admin(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    pid = (data.get("participant_id") or "").strip().upper()
    group = (data.get("group") or "").strip().lower()
    if not pid.startswith("P") or not pid[1:].isdigit() or len(pid[1:]) < 3:
        return JSONResponse(
            {"error": "participant_id must be P9999 style (at least 3 digits, with prefix P)"},
            status_code=400,
        )
    if group not in ("exp", "ctrl"):
        return JSONResponse({"error": "group must be 'exp' or 'ctrl'"}, status_code=400)

    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute(
        "SELECT is_admin FROM participants WHERE participant_id=?", (pid,)
    ).fetchone()
    if existing and not (existing[0] or 0):
        conn.close()
        return JSONResponse(
            {"error": f"participant_id '{pid}' already exists as a research participant"},
            status_code=409,
        )

    now = time.time()
    if existing:
        # Admin id is intentionally reusable for QA; refresh the selected arm.
        conn.execute(
            "UPDATE participants SET group_type=?, assigned_at=?, is_admin=1, source='admin' "
            "WHERE participant_id=?",
            (group, now, pid),
        )
    else:
        conn.execute(
            "INSERT INTO participants (participant_id, created_at, group_type, pre_total, assigned_at, is_admin, source) "
            "VALUES (?,?,?,?,?,?,?)",
            (pid, now, group, None, now, 1, "admin"),
        )
    conn.commit(); conn.close()
    return {"participant_id": pid, "group": group, "is_admin": True, "reusable": True}


# ──────────────────────────────────────────────────────────────────────
# POST /api/assign
# Body (JSON): { participant_id, pre_total }
# Stratified randomization by pre_total; within a stratum assign to the
# smaller arm, ties broken randomly. Persists the assignment.
# Returns: { group: "exp"|"ctrl", participant_id }
# ──────────────────────────────────────────────────────────────────────
def _stratum(pre_total: int) -> str:
    return "low" if pre_total <= 10 else "high"

@app.post("/api/assign")
async def assign(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    pid = data.get("participant_id")
    if not pid:
        return JSONResponse({"error": "participant_id required"}, status_code=400)
    try:
        pre_total = int(data.get("pre_total", 0))
    except (TypeError, ValueError):
        pre_total = 0

    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT group_type, is_admin FROM participants WHERE participant_id=?", (pid,)
    ).fetchone()
    if not row:
        conn.close()
        return JSONResponse({"error": "participant not found"}, status_code=404)

    current_group, is_admin = row[0], row[1] or 0

    # Idempotent: if already assigned (incl. admin with manual group), return existing.
    if current_group in ("exp", "ctrl"):
        conn.close()
        return {"group": current_group, "participant_id": pid, "is_admin": bool(is_admin)}

    stratum = _stratum(pre_total)
    # Count already-assigned participants in the same stratum.
    n_exp = n_ctrl = 0
    for g, pt in conn.execute(
        "SELECT group_type, pre_total FROM participants "
        "WHERE group_type IN ('exp','ctrl') AND pre_total IS NOT NULL"
    ).fetchall():
        if _stratum(pt) != stratum:
            continue
        if g == "exp":
            n_exp += 1
        elif g == "ctrl":
            n_ctrl += 1

    if n_exp < n_ctrl:
        group_type = "exp"
    elif n_ctrl < n_exp:
        group_type = "ctrl"
    else:
        group_type = random.choice(["exp", "ctrl"])

    conn.execute(
        "UPDATE participants SET group_type=?, pre_total=?, assigned_at=? WHERE participant_id=?",
        (group_type, pre_total, time.time(), pid),
    )
    conn.commit(); conn.close()
    return {"group": group_type, "participant_id": pid}


@app.get("/api/delayed/{participant_id}")
async def delayed(participant_id: str):
    pid = (participant_id or "").strip().upper()
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT group_type, is_admin FROM participants WHERE participant_id=?", (pid,)
    ).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "participant not found"}, status_code=404)
    group_type, is_admin = row[0], row[1] or 0
    if group_type not in ("exp", "ctrl"):
        return JSONResponse({"error": "participant chưa hoàn tất pre/assign"}, status_code=409)
    return {"participant_id": pid, "group": group_type, "is_admin": bool(is_admin)}


# ──────────────────────────────────────────────────────────────────────
# POST /api/session/complete
# Body (JSON): group, preScore, postScore, practiceScore, attempts[],
#              participant_id, phase, paas{germane,extraneous}
#   attempts = [{qIdx, tense, correct, attempts, wrongWord, voiceText}]
# Returns: { session_id }
# ──────────────────────────────────────────────────────────────────────
def _opt_int(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None

@app.post("/api/session/complete")
async def session_complete(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    paas = data.get("paas", {}) or {}
    sid = str(uuid.uuid4())[:8]
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions "
        "(id, created_at, group_type, pre_score, post_score, practice_score, "
        " attempts, debrief, participant_id, phase, paas_germane, paas_extraneous) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            sid,
            time.time(),
            data.get("group", "exp"),
            int(data.get("preScore", 0)),
            int(data.get("postScore", 0)),
            int(data.get("practiceScore", 0)),
            json.dumps(data.get("attempts", [])),
            None,   # debrief generated on demand
            data.get("participant_id"),
            data.get("phase", "post"),
            _opt_int(paas.get("germane")),
            _opt_int(paas.get("extraneous")),
        )
    )
    conn.commit(); conn.close()
    return {"session_id": sid}


# ──────────────────────────────────────────────────────────────────────
# GET /api/debrief/{session_id}
# Streams the post-session mentor debrief via DeepSeek API as SSE
# (text/event-stream), then caches it in SQLite.
# ──────────────────────────────────────────────────────────────────────
def _build_debrief_prompt(row: dict) -> str:
    attempts = json.loads(row["attempts"] or "[]")

    wrong = [a for a in attempts if not a.get("correct")]
    right = [a for a in attempts if a.get("correct")]

    wrong_lines = "\n".join(
        f'  - "{a.get("tense","?")}" — chose "{a.get("wrongWord","?")}"'
        + (f', voice said: "{a.get("voiceText","")}"' if a.get("voiceText") else "")
        + f', needed {a.get("attempts",1)} attempt(s)'
        for a in wrong
    ) or "  (none)"

    right_lines = "\n".join(
        f'  - "{a.get("tense","?")}"' for a in right
    ) or "  (none)"

    pre  = row["pre_score"]
    post = row["post_score"]
    pts  = row["practice_score"]
    total = 20
    grp  = "Experimental (spoke aloud)" if row["group_type"] == "exp" else "Control (silent recall)"

    return f"""You are a warm but precise TOEIC grammar coach writing a post-session debrief.

SESSION DATA:
- Group: {grp}
- Pre-test: {pre}/{total} correct
- Post-test: {post}/{total} correct  
- Practice score: {pts} pts
- Pre→Post delta: {'+' if post >= pre else ''}{post - pre} questions

WRONG during practice:
{wrong_lines}

CORRECT during practice:
{right_lines}

Write the debrief in Vietnamese for a Vietnamese university student preparing for TOEIC. Keep English grammar terms when useful.

Write a concise personal debrief (max 180 Vietnamese words) with these 4 parts:
1. **Bạn làm tốt ở đâu** — 1 sentence acknowledging strengths
2. **Bạn trượt ở đâu** — name each grammar rule that caused errors, explain WHY it's tricky in 1 line each
3. **Khoảng cách quan trọng** — 1 insight about what the pre→post delta tells us
4. **Việc cần làm ngày mai** — one specific practice action (e.g. "Viết 3 câu dùng 'a number of + plural verb' trong email công việc")

Tone: direct, encouraging, student-friendly, no fluff. Use short paragraphs.
Do NOT start with generic praise like "Làm tốt lắm"."""


async def _stream_debrief(prompt: str, session_id: str):
    """Stream mentor debrief via DeepSeek API."""
    # Check cache first
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT debrief FROM sessions WHERE id=?", (session_id,)).fetchone()
    conn.close()

    if row and row[0]:
        # Cached — stream from cache instantly
        cached = row[0]
        for chunk in [cached[i:i+80] for i in range(0, len(cached), 80)]:
            yield f"data: {json.dumps({'text': chunk})}\n\n"
            await __import__('asyncio').sleep(0.02)
        yield "data: [DONE]\n\n"
        return

    # Call DeepSeek API. The key must come from process env; never hardcode secrets.
    if not DEEPSEEK_API_KEY:
        yield f"data: {json.dumps({'text': 'Debrief unavailable: DEEPSEEK_API_KEY is not configured.'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    try:
        resp = requests.post(
            DEEPSEEK_CHAT_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": "Bạn là một TOEIC mentor thân thiện, trực tiếp, khuyến khích học sinh Việt Nam."},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 800,
            },
            stream=True,
            timeout=60,
        )
        full_text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8", errors="ignore").strip()
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                break
            try:
                obj = json.loads(line)
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    full_text += delta
                    yield f"data: {json.dumps({'text': delta})}\n\n"
            except Exception:
                pass

        clean = full_text.strip()

        # Cache it
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE sessions SET debrief=? WHERE id=?", (clean, session_id))
        conn.commit(); conn.close()

        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'text': f'Mentor error: {e}'})}\n\n"
        yield "data: [DONE]\n\n"


@app.get("/api/status/{participant_id}")
async def get_status(participant_id: str):
    """Return the latest completed session for a participant (for resume logic)."""
    conn = sqlite3.connect(DB_PATH)
    p = conn.execute(
        "SELECT group_type FROM participants WHERE participant_id=?", (participant_id,)
    ).fetchone()
    if not p:
        conn.close()
        return JSONResponse({"error": "not found"}, status_code=404)
    row = conn.execute(
        "SELECT id, pre_score, post_score, practice_score, phase FROM sessions "
        "WHERE participant_id=? ORDER BY created_at DESC LIMIT 1",
        (participant_id,)
    ).fetchone()
    conn.close()
    if not row:
        return {"phase": "pre", "group": p[0], "session_id": None}
    return {
        "phase": row[4] or "post",
        "group": p[0],
        "session_id": row[0],
        "pre_score": row[1],
        "post_score": row[2],
        "practice_score": row[3],
    }


@app.get("/api/debrief_text/{session_id}")
async def get_debrief_text(session_id: str):
    """Return cached debrief as JSON for browsers/proxies that fail SSE."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT debrief FROM sessions WHERE id=?", (session_id,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    if not row[0]:
        return JSONResponse({"error": "Debrief not ready"}, status_code=404)
    return {"text": row[0]}


@app.get("/api/debrief/{session_id}")
async def get_debrief(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT id,group_type,pre_score,post_score,practice_score,attempts,debrief FROM sessions WHERE id=?",
        (session_id,)
    ).fetchone()
    conn.close()

    if not row:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    rowdict = {
        "id": row[0], "group_type": row[1], "pre_score": row[2],
        "post_score": row[3], "practice_score": row[4],
        "attempts": row[5], "debrief": row[6],
    }
    prompt = _build_debrief_prompt(rowdict)
    return StreamingResponse(
        _stream_debrief(prompt, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

