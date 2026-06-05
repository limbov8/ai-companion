from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@dataclass
class ConversationLogger:
    path: Path = Path("logs/conversations.log")

    def record(self, session_id: str, role: str, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).isoformat()
        line = f"{stamp}\t{session_id}\t{role}\t{text.replace(chr(9), ' ')}\n"
        self.path.write_text(
            self.path.read_text(encoding="utf-8") + line if self.path.exists() else line,
            encoding="utf-8",
        )
