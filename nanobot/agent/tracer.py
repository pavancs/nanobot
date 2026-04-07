"""Debug tracer for logging detailed message processing traces."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

MAX_TRACES = 1000


@dataclass
class LLMCallTrace:
    iteration: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    duration_ms: int = 0
    response_type: str = ""
    content_preview: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolTrace:
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result_preview: str = ""
    result_tokens: int = 0
    duration_ms: int = 0
    status: str = ""


@dataclass
class Trace:
    trace_id: str = ""
    timestamp: str = ""
    channel: str = ""
    sender_id: str = ""
    session_key: str = ""
    input_message: str = ""
    system_prompt_tokens: int = 0
    system_prompt_sections: list[str] = field(default_factory=list)
    history_messages: int = 0
    history_tokens: int = 0
    total_input_messages: int = 0
    llm_calls: list[LLMCallTrace] = field(default_factory=list)
    tool_executions: list[ToolTrace] = field(default_factory=list)
    model: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration_ms: int = 0
    final_response: str = ""
    stop_reason: str = ""
    error: str | None = None
    _start_time: float = field(default=0.0, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "channel": self.channel,
            "sender_id": self.sender_id,
            "session_key": self.session_key,
            "input_message": self.input_message,
            "context": {
                "system_prompt_tokens": self.system_prompt_tokens,
                "system_prompt_sections": self.system_prompt_sections,
                "history_messages": self.history_messages,
                "history_tokens": self.history_tokens,
                "total_input_messages": self.total_input_messages,
            },
            "model": self.model,
            "llm_calls": [
                {
                    "iteration": c.iteration,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cached_tokens": c.cached_tokens,
                    "duration_ms": c.duration_ms,
                    "response_type": c.response_type,
                    "content_preview": c.content_preview,
                    "tool_calls": c.tool_calls,
                }
                for c in self.llm_calls
            ],
            "tool_executions": [
                {
                    "name": t.name,
                    "args": t.args,
                    "result_preview": t.result_preview,
                    "result_tokens": t.result_tokens,
                    "duration_ms": t.duration_ms,
                    "status": t.status,
                }
                for t in self.tool_executions
            ],
            "summary": {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_duration_ms": self.total_duration_ms,
                "stop_reason": self.stop_reason,
                "error": self.error,
            },
            "final_response": self.final_response[:500] if self.final_response else "",
        }


class Tracer:
    def __init__(self, workspace: Path):
        self.traces_dir = workspace / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.stats_file = self.traces_dir / "stats.jsonl"
        self._active: Trace | None = None

    @property
    def active(self) -> Trace | None:
        return self._active

    def start(self, channel: str, sender_id: str, session_key: str, input_message: str, model: str = "") -> Trace:
        now = datetime.now()
        trace = Trace(
            trace_id=now.strftime("%Y%m%d_%H%M%S") + f"_{id(now) % 10000:04d}",
            timestamp=now.isoformat(),
            channel=channel,
            sender_id=sender_id,
            session_key=session_key,
            input_message=input_message[:500],
            model=model,
            _start_time=time.monotonic(),
        )
        self._active = trace
        return trace

    def log_context(self, system_prompt_tokens: int, sections: list[str], history_messages: int, history_tokens: int, total_messages: int) -> None:
        if not self._active:
            return
        self._active.system_prompt_tokens = system_prompt_tokens
        self._active.system_prompt_sections = sections
        self._active.history_messages = history_messages
        self._active.history_tokens = history_tokens
        self._active.total_input_messages = total_messages

    def log_llm_call(self, iteration: int, model: str, usage: dict[str, int], duration_ms: int, response_type: str, content_preview: str = "", tool_calls: list[dict[str, Any]] | None = None) -> None:
        if not self._active:
            return
        call = LLMCallTrace(
            iteration=iteration, model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cached_tokens=usage.get("cached_tokens", 0),
            duration_ms=duration_ms, response_type=response_type,
            content_preview=content_preview[:300] if content_preview else "",
            tool_calls=tool_calls or [],
        )
        self._active.llm_calls.append(call)

    def log_tool(self, name: str, args: dict[str, Any], result_preview: str, duration_ms: int, status: str) -> None:
        if not self._active:
            return
        safe_args = {}
        for k, v in (args or {}).items():
            sv = str(v)
            safe_args[k] = sv[:200] if len(sv) > 200 else sv
        tool = ToolTrace(
            name=name, args=safe_args,
            result_preview=result_preview[:300] if result_preview else "",
            result_tokens=len(result_preview) // 4 if result_preview else 0,
            duration_ms=duration_ms, status=status,
        )
        self._active.tool_executions.append(tool)

    def finish(self, final_response: str | None = None, usage: dict[str, int] | None = None, stop_reason: str = "completed", error: str | None = None) -> Trace | None:
        if not self._active:
            return None
        trace = self._active
        trace.total_duration_ms = int((time.monotonic() - trace._start_time) * 1000)
        trace.final_response = final_response or ""
        trace.stop_reason = stop_reason
        trace.error = error
        if usage:
            trace.total_input_tokens = usage.get("prompt_tokens", 0)
            trace.total_output_tokens = usage.get("completion_tokens", 0)
        else:
            trace.total_input_tokens = sum(c.input_tokens for c in trace.llm_calls)
            trace.total_output_tokens = sum(c.output_tokens for c in trace.llm_calls)
        self._save_trace(trace)
        self._append_stats(trace)
        self._cleanup_old_traces()
        self._active = None
        return trace

    def _save_trace(self, trace: Trace) -> None:
        try:
            filename = f"{trace.trace_id}_{trace.channel}_{trace.sender_id}.json"
            filepath = self.traces_dir / filename
            filepath.write_text(json.dumps(trace.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save trace: {}", e)

    def _append_stats(self, trace: Trace) -> None:
        try:
            stat = {
                "date": trace.timestamp[:10],
                "time": trace.timestamp[11:19],
                "tokens_in": trace.total_input_tokens,
                "tokens_out": trace.total_output_tokens,
                "model": trace.model,
                "duration_ms": trace.total_duration_ms,
                "tools_used": [t.name for t in trace.tool_executions],
                "channel": trace.channel,
                "stop_reason": trace.stop_reason,
            }
            with open(self.stats_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(stat, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to append stats: {}", e)

    def _cleanup_old_traces(self) -> None:
        try:
            trace_files = sorted(
                [f for f in self.traces_dir.iterdir() if f.suffix == ".json"],
                key=lambda f: f.stat().st_mtime,
            )
            if len(trace_files) > MAX_TRACES:
                for f in trace_files[:len(trace_files) - MAX_TRACES]:
                    f.unlink()
        except Exception as e:
            logger.warning("Failed to cleanup old traces: {}", e)
