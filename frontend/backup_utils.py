"""Bounded, per-file backups made immediately before CSV writes."""

from datetime import datetime
from pathlib import Path
import shutil
from uuid import uuid4


def create_backup(csv_path, keep=10):
    """Copy an existing CSV into its sibling backups directory, then prune its own history."""
    source = Path(csv_path)
    if not source.exists():
        return None
    backup_dir = source.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest = backup_dir / f"{source.stem}_backup_{stamp}_{uuid4().hex[:8]}{source.suffix}"
    shutil.copy2(source, dest)
    backups = sorted(backup_dir.glob(f"{source.stem}_backup_*{source.suffix}"),
                     key=lambda path: (path.stat().st_mtime_ns, path.name))
    for old in backups[:-keep]:
        old.unlink()
    return dest
