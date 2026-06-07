"""Content-addressed cache for code-gen artifacts.

Cache key = SHA-256 of (schema source bytes + user prompt).
- File-backed schemas:  hashes the raw file bytes  → stable across environments
- Dict / DB-inferred:   hashes JSON-serialised dict → invalidates on any column change
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union


def compute_schema_hash(schema_source: Union[str, dict, type], user_prompt: str = "") -> str:
    """Return a 16-char hex hash that identifies this schema + prompt combination.

    For file paths the file content is hashed, so the same schema file used in
    dev/staging/prod all produce the same hash regardless of table name.
    """
    if isinstance(schema_source, str) and os.path.exists(schema_source):
        with open(schema_source, "rb") as f:
            content = f.read()
    elif isinstance(schema_source, dict):
        content = json.dumps(schema_source, sort_keys=True, default=str).encode()
    else:
        # SQLAlchemy model or other type — use repr as best-effort stable key
        content = repr(schema_source).encode()

    digest = hashlib.sha256(content + user_prompt.encode()).hexdigest()
    return digest[:16]


class CodegenCache:
    """Read/write code-gen artifacts keyed by schema content hash.

    Files are named ``{table_name}_{hash[:8]}.json`` so the cache directory is
    immediately human-readable.  At lookup time only the hash is known, so
    ``load()`` globs for ``*_{hash}.json``.
    """

    def __init__(self, cache_dir: str = ".syda_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _filename(self, schema_hash: str, hint_table_name: str = "") -> str:
        prefix = f"{hint_table_name}_" if hint_table_name else ""
        return f"{prefix}{schema_hash}.json"

    def load(self, schema_hash: str) -> Optional[dict]:
        """Return the cached artifact dict or None if not found / unreadable.

        Globs for ``*{hash}.json`` so it works regardless of what table-name
        prefix was used when the artifact was saved.
        """
        import glob
        pattern = os.path.join(self.cache_dir, f"*{schema_hash}.json")
        matches = glob.glob(pattern)
        if not matches:
            return None
        try:
            with open(matches[0]) as f:
                return json.load(f)
        except Exception:
            return None

    def save(
        self,
        schema_hash: str,
        simple: Dict[str, str],
        semantic: List[str],
        model: str,
        hint_table_name: str = "",
    ) -> None:
        """Persist a code-gen artifact, replacing any previous version for this table."""
        import glob

        # Remove stale artifacts for the same table before writing the new one
        # so only one file per table exists at any time (dbt-style overwrite).
        if hint_table_name:
            stale = glob.glob(os.path.join(self.cache_dir, f"{hint_table_name}_*.json"))
            for f in stale:
                try:
                    os.remove(f)
                except OSError:
                    pass

        artifact = {
            "schema_hash": schema_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "hint_table_name": hint_table_name,
            "simple": simple,
            "semantic": semantic,
        }
        path = os.path.join(self.cache_dir, self._filename(schema_hash, hint_table_name))
        with open(path, "w") as f:
            json.dump(artifact, f, indent=2)
