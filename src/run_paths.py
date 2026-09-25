"""Create a fresh directory for each experiment without overwriting past runs."""

import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def create_run_directory(base, name=None):
    if name is None:
        name = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Run name must be 1-80 letters, digits, underscores or hyphens, starting with a letter or digit")
    if name.split("_")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("Run name uses a reserved Windows device name")
    destination = Path(base) / name
    destination.mkdir(parents=True, exist_ok=False)
    return destination
