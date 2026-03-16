from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Optional
from urllib import request


DEFAULT_DISCORD_USERNAME = "SmartAirbase Trainer"


def resolve_discord_webhook_url(explicit_url: Optional[str] = None) -> Optional[str]:
    if explicit_url:
        return explicit_url
    return (
        os.getenv("DISCORD_WEBHOOK_URL")
        or os.getenv("DISCORD_WEBHOOK")
        or None
    )


@dataclass(slots=True)
class DiscordNotifier:
    webhook_url: Optional[str]
    username: str = DEFAULT_DISCORD_USERNAME
    timeout_seconds: float = 10.0

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send(self, content: str) -> bool:
        if not self.enabled:
            return False

        payload = {
            "content": content[:2000],
            "username": self.username,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.webhook_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "SmartAirbaseTrainer/1.0",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                response.read()
            return True
        except Exception as exc:  # pragma: no cover - network failure path
            print(f"[discord] notification failed: {exc}", file=sys.stderr)
            return False
