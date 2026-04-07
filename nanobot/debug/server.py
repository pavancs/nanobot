"""Debug trace viewer — lightweight FastAPI server."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Nanobot Debug Viewer")

TRACES_DIR: Path | None = None


def set_traces_dir(path: Path) -> None:
    global TRACES_DIR
    TRACES_DIR = path


def _get_traces_dir() -> Path:
    if TRACES_DIR:
        return TRACES_DIR
    return Path.home() / ".nanobot" / "workspace" / "traces"


@app.get("/", response_class=HTMLResponse)
async def viewer():
    html_path = Path(__file__).parent / "viewer.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/traces")
async def list_traces(limit: int = 50, offset: int = 0):
    traces_dir = _get_traces_dir()
    if not traces_dir.exists():
        return []
    files = sorted(
        [f for f in traces_dir.iterdir() if f.suffix == ".json"],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    result = []
    for f in files[offset : offset + limit]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            result.append({
                "trace_id": data.get("trace_id", ""),
                "timestamp": data.get("timestamp", ""),
                "channel": data.get("channel", ""),
                "sender_id": data.get("sender_id", ""),
                "input_message": data.get("input_message", "")[:100],
                "model": data.get("model", ""),
                "summary": data.get("summary", {}),
                "llm_calls_count": len(data.get("llm_calls", [])),
                "tools_count": len(data.get("tool_executions", [])),
                "filename": f.name,
            })
        except Exception:
            continue
    return result


@app.get("/api/traces/{filename}")
async def get_trace(filename: str):
    traces_dir = _get_traces_dir()
    filepath = traces_dir / filename
    if not filepath.exists() or not filepath.suffix == ".json":
        return JSONResponse({"error": "not found"}, status_code=404)
    return json.loads(filepath.read_text(encoding="utf-8"))


@app.get("/api/stats")
async def get_stats():
    traces_dir = _get_traces_dir()
    stats_file = traces_dir / "stats.jsonl"
    if not stats_file.exists():
        return {"today": {}, "week": {}, "month": {}, "all_time": {}}

    entries = []
    for line in stats_file.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except Exception:
                continue

    today = datetime.now().strftime("%Y-%m-%d")
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    def summarize(filtered):
        if not filtered:
            return {"messages": 0, "tokens_in": 0, "tokens_out": 0, "avg_duration_ms": 0}
        tokens_in = sum(e.get("tokens_in", 0) for e in filtered)
        tokens_out = sum(e.get("tokens_out", 0) for e in filtered)
        durations = [e.get("duration_ms", 0) for e in filtered]
        return {
            "messages": len(filtered),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "avg_duration_ms": int(sum(durations) / len(durations)) if durations else 0,
        }

    return {
        "today": summarize([e for e in entries if e.get("date") == today]),
        "week": summarize([e for e in entries if e.get("date", "") >= week_ago]),
        "month": summarize([e for e in entries if e.get("date", "") >= month_ago]),
        "all_time": summarize(entries),
    }


@app.get("/api/stats/daily")
async def get_daily_stats(days: int = 30):
    traces_dir = _get_traces_dir()
    stats_file = traces_dir / "stats.jsonl"
    if not stats_file.exists():
        return []

    daily = defaultdict(lambda: {"messages": 0, "tokens_in": 0, "tokens_out": 0, "duration_ms": 0})
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    for line in stats_file.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
            date = e.get("date", "")
            if date < cutoff:
                continue
            daily[date]["messages"] += 1
            daily[date]["tokens_in"] += e.get("tokens_in", 0)
            daily[date]["tokens_out"] += e.get("tokens_out", 0)
            daily[date]["duration_ms"] += e.get("duration_ms", 0)
        except Exception:
            continue

    return [
        {"date": date, **vals}
        for date, vals in sorted(daily.items())
    ]


def run_server(host: str = "0.0.0.0", port: int = 18791):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
