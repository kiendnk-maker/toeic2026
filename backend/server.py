"""
standup_combined/server.py
Serves static files from /var/www/mailscanner/standup/
AND proxies /api/* to standup-backend on :8090
Single process — no nginx reload needed.
"""
import asyncio, pathlib, mimetypes, httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

STATIC_DIR = pathlib.Path("/var/www/mailscanner/standup")
BACKEND_URL = "http://127.0.0.1:8090"

app = FastAPI(docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def no_cache_index(request: Request, call_next):
    response = await call_next(request)
    if request.url.path in ("/", "/index.html"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response

_client = httpx.AsyncClient(base_url=BACKEND_URL, timeout=35.0)

@app.api_route("/api/{path:path}", methods=["GET","POST","PUT","DELETE","PATCH","OPTIONS"])
async def proxy_api(path: str, request: Request):
    """Transparent reverse proxy to the standup FastAPI backend."""
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    resp = await _client.request(
        method=request.method,
        url=f"/api/{path}",
        params=dict(request.query_params),
        headers=headers,
        content=body,
    )
    return Response(content=resp.content, status_code=resp.status_code,
                    headers=dict(resp.headers), media_type=resp.headers.get("content-type"))

# Static files — must come AFTER the /api/ route
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
