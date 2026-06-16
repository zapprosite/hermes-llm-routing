"""LlmRoutingPlugin: hermes-llm-routing para hermes-agent."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("hermes-llm-routing")


class LlmRoutingPlugin:
    """Plugin backend."""
    name = "hermes-llm-routing"
    kind = "backend"
    version = "1.0.0"

    def register(self, ctx) -> None:
        """Hook de registro."""
        # Tools
        ctx.register_tool("hermes_llm_routing_status", self._tool_status)

        # Skills
        skill_path = self._skill_path()
        if skill_path.exists():
            ctx.register_skill("hermes-llm-routing", skill_path)

        log.info("hermes-llm-routing v%s registrado", self.version)

    def _skill_path(self) -> Path:
        return Path(__file__).parent.parent.parent / "skills" / "llm-routing"

    def _tool_status(self, **_):
        return {"status": "ready", "version": self.version}
