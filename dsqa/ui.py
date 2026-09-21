from __future__ import annotations

import json
import mimetypes
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dsqa.config import CONFIG

_STATIC = Path(__file__).with_name("static")


def _unique_sources(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    out: list[dict[str, Any]] = []
    for doc in docs:
        key = (str(doc.get("source", "")), int(doc.get("page") or 0))
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": key[0], "page": key[1]})
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def state_payload(state: dict[str, Any]) -> dict[str, Any]:
    docs = list(state.get("docs") or [])
    answer = str(state.get("answer") or "")
    spans = [_json_safe(s) for s in (state.get("spans") or [])]
    return {
        "answer": answer,
        "grounded": None if not CONFIG.grounding_check else state.get("grounded"),
        "reason": state.get("reason") or "",
        "attempts": int(state.get("attempts") or 0),
        "search_query": state.get("search_query") or "",
        "abstained": "INSUFFICIENT_CONTEXT" in answer,
        "sources": _unique_sources(docs),
        "docs": [
            {
                "source": d.get("source"),
                "page": d.get("page"),
                "text": d.get("text"),
                "distance": _json_safe(d.get("distance")),
                "rerank": _json_safe(d.get("rerank")),
            }
            for d in docs
        ],
        "trace": list(state.get("trace") or []),
        "spans": spans,
        "cost_usd": round(sum(float(s.get("cost_usd") or 0) for s in spans), 6),
        "latency_ms": round(sum(float(s.get("ms") or 0) for s in spans), 1),
    }


def _ask_payload(question: str) -> dict[str, Any]:
    from dsqa.graph import ask

    return state_payload(dict(ask(question)))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self._send(code, raw, "application/json; charset=utf-8")

    def _sse_write(self, event: str, data: Any) -> None:
        payload = json.dumps(_json_safe(data), default=str)
        self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream_ask(self, question: str) -> None:
        from dsqa.graph import ask_stream

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for event in ask_stream(question):
                name = str(event.get("event") or "message")
                data = event.get("data")
                if name == "done" and isinstance(data, dict):
                    data = state_payload(data)
                self._sse_write(name, data)
        except Exception as exc:
            self._sse_write("error", {"error": str(exc)})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            html = (_STATIC / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
            return
        if path == "/api/stats":
            from dsqa.store import count

            self._json(200, {"chunks": count(), "config": asdict(CONFIG)})
            return
        if path.startswith("/static/"):
            rel = path.removeprefix("/static/")
            target = (_STATIC / rel).resolve()
            if not target.is_relative_to(_STATIC.resolve()) or not target.is_file():
                self._json(404, {"error": "not found"})
                return
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if target.suffix == ".js":
                ctype = "application/javascript; charset=utf-8"
            elif target.suffix == ".css":
                ctype = "text/css; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/ask":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return
        question = str(body.get("question") or "").strip()
        if not question:
            self._json(400, {"error": "question is required"})
            return
        if len(question) > CONFIG.max_question_chars:
            self._json(400, {"error": f"question exceeds {CONFIG.max_question_chars} characters"})
            return
        accept = (self.headers.get("Accept") or "").lower()
        stream = bool(body.get("stream")) or "text/event-stream" in accept
        try:
            if stream:
                self._stream_ask(question)
                return
            self._json(200, _ask_payload(question))
        except Exception as exc:
            self._json(500, {"error": str(exc)})


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"dsqa UI → http://{host}:{port}")
    httpd.serve_forever()
