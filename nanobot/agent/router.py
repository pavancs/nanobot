"""Message router — classifies messages and picks the right model/subagent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class RoutingDecision:
    route: str = "direct"           # "direct" or "subagent"
    subagent: str = ""              # subagent name (e.g. "thinker", "coder")
    model: str = ""                 # model to use
    reason: str = ""               # why this route was chosen
    method: str = ""               # "keyword", "heuristic", or "llm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "subagent": self.subagent,
            "model": self.model,
            "reason": self.reason,
            "method": self.method,
        }


class Router:
    """Classifies messages and picks the right model/subagent."""

    def __init__(self, provider: Any, router_config: Any, subagents_config: dict, default_model: str):
        self.provider = provider
        self.router_config = router_config
        self.subagents = subagents_config
        self.default_model = default_model
        self.enabled = router_config.enabled if router_config else False
        self.classify_model = (router_config.model if router_config and router_config.model else default_model)

    _GREETING_PATTERNS = [
        "hi", "hello", "hey", "good morning", "good evening", "good night",
        "good afternoon", "howdy", "sup", "yo", "hola", "namaste",
        "thanks", "thank you", "bye", "goodbye", "see you", "gn", "gm",
        "how are you", "what's up", "whats up",
    ]

    async def classify(self, message: str) -> RoutingDecision:
        if not self.enabled:
            return RoutingDecision(route="direct", model=self.default_model, reason="router disabled", method="none")

        msg_lower = message.lower().strip()

        # Stage 1: Greetings handled directly (no subagent needed)
        for pattern in self._GREETING_PATTERNS:
            if msg_lower == pattern or msg_lower.startswith(pattern + " ") or msg_lower.endswith(" " + pattern):
                return RoutingDecision(
                    route="direct",
                    model=self.default_model,
                    reason=f"greeting: {pattern}",
                    method="greeting",
                )

        # Stage 2: Keyword matching to specific subagent
        for name, cfg in self.subagents.items():
            triggers = cfg.triggers if hasattr(cfg, "triggers") else cfg.get("triggers", [])
            model = cfg.model if hasattr(cfg, "model") else cfg.get("model", "")
            for trigger in triggers:
                if trigger.lower() in msg_lower:
                    return RoutingDecision(
                        route="subagent",
                        subagent=name,
                        model=model,
                        reason=f"keyword: {trigger}",
                        method="keyword",
                    )

        # Stage 3: Everything else → default subagent (thinker)
        # Orchestrator always delegates non-greeting messages
        default_subagent = "thinker"
        if default_subagent in self.subagents:
            cfg = self.subagents[default_subagent]
            model = cfg.model if hasattr(cfg, "model") else cfg.get("model", self.default_model)
            return RoutingDecision(
                route="subagent",
                subagent=default_subagent,
                model=model,
                reason="default delegation",
                method="default",
            )

        return RoutingDecision(route="direct", model=self.default_model, reason="no subagents configured", method="fallback")

    async def _llm_classify(self, message: str) -> RoutingDecision:
        subagent_list = ", ".join(
            f"{name} ({cfg.description if hasattr(cfg, 'description') else cfg.get('description', '')})"
            for name, cfg in self.subagents.items()
        )
        prompt = f"""Classify this message into one category. Reply with ONLY the category name.

Categories:
- direct: greetings, quick questions, simple facts, conversions, casual chat
{chr(10).join(f'- {name}: {cfg.description if hasattr(cfg, description) else cfg.get(description, )}' for name, cfg in self.subagents.items())}

Message: {message[:300]}"""

        response = await self.provider.chat_with_retry(
            model=self.classify_model,
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            max_tokens=20,
        )
        answer = (response.content or "direct").strip().lower().split()[0]

        if answer in self.subagents:
            cfg = self.subagents[answer]
            model = cfg.model if hasattr(cfg, 'model') else cfg.get('model', self.default_model)
            return RoutingDecision(
                route="subagent",
                subagent=answer,
                model=model,
                reason=f"llm classified as {answer}",
                method="llm",
            )

        return RoutingDecision(route="direct", model=self.default_model, reason=f"llm: {answer}", method="llm")
