from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping

import torch


CACHE_FORMAT_VERSION = 1


def text_cache_key(text: str) -> str:
    """Hash the exact prompt passed to the tokenizer."""
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _tensor_to_blob(tensor: torch.Tensor) -> tuple[str, bytes]:
    tensor = tensor.detach().contiguous().cpu()
    if tensor.dtype == torch.bfloat16:
        return "bfloat16", tensor.view(torch.uint16).numpy().tobytes()
    if tensor.dtype == torch.float16:
        return "float16", tensor.numpy().tobytes()
    if tensor.dtype == torch.float32:
        return "float32", tensor.numpy().tobytes()
    raise TypeError(f"Unsupported cache dtype: {tensor.dtype}")


def _blob_to_tensor(blob: bytes, dtype_name: str, shape: tuple[int, ...]) -> torch.Tensor:
    if dtype_name == "bfloat16":
        tensor = torch.frombuffer(
            bytearray(blob), dtype=torch.uint16
        ).clone().view(torch.bfloat16)
    elif dtype_name == "float16":
        tensor = torch.frombuffer(
            bytearray(blob), dtype=torch.float16
        ).clone()
    elif dtype_name == "float32":
        tensor = torch.frombuffer(
            bytearray(blob), dtype=torch.float32
        ).clone()
    else:
        raise ValueError(f"Unsupported cached dtype: {dtype_name}")
    return tensor.reshape(shape)


class TextFeatureCache:
    """SQLite-backed, read-mostly cache for frozen LLM token features."""

    def __init__(self, path: str | Path, read_only: bool = True):
        self.path = Path(path).expanduser().resolve()
        if read_only and not self.path.is_file():
            raise FileNotFoundError(f"Text feature cache not found: {self.path}")
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        if read_only:
            uri = f"{self.path.as_uri()}?mode=ro"
            self.connection = sqlite3.connect(uri, uri=True, timeout=60)
        else:
            self.connection = sqlite3.connect(str(self.path), timeout=60)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self._create_schema()
        self.read_only = read_only
        self.metadata = self._read_metadata()
        if self.metadata:
            version = int(self.metadata.get("format_version", -1))
            if version != CACHE_FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported text cache format {version}; "
                    f"expected {CACHE_FORMAT_VERSION}."
                )

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                name TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS features (
                cache_key TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                seq_len INTEGER NOT NULL,
                hidden_dim INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                data BLOB NOT NULL
            );
            """
        )
        self.connection.commit()

    def _read_metadata(self) -> dict[str, object]:
        try:
            rows = self.connection.execute(
                "SELECT name, value FROM metadata"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            raise ValueError(f"Invalid text feature cache: {self.path}") from exc
        return {name: json.loads(value) for name, value in rows}

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        if self.read_only:
            raise RuntimeError("Cannot update a read-only text feature cache.")
        values = dict(metadata)
        values["format_version"] = CACHE_FORMAT_VERSION
        self.connection.executemany(
            "INSERT OR REPLACE INTO metadata(name, value) VALUES (?, ?)",
            [(name, json.dumps(value, ensure_ascii=False)) for name, value in values.items()],
        )
        self.connection.commit()
        self.metadata = values

    def contains(self, text: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM features WHERE cache_key=?",
            (text_cache_key(text),),
        ).fetchone()
        return row is not None

    def put(self, text: str, valid_features: torch.Tensor) -> None:
        if self.read_only:
            raise RuntimeError("Cannot update a read-only text feature cache.")
        if valid_features.ndim != 2 or valid_features.shape[0] < 1:
            raise ValueError(
                "Expected non-empty [valid_tokens, hidden_dim] features."
            )
        dtype_name, blob = _tensor_to_blob(valid_features)
        prompt = str(text or "")
        self.connection.execute(
            """
            INSERT OR REPLACE INTO features(
                cache_key, prompt, seq_len, hidden_dim, dtype, data
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                text_cache_key(prompt),
                prompt,
                int(valid_features.shape[0]),
                int(valid_features.shape[1]),
                dtype_name,
                sqlite3.Binary(blob),
            ),
        )

    def commit(self) -> None:
        if not self.read_only:
            self.connection.commit()

    def get(self, text: str) -> torch.Tensor:
        prompt = str(text or "")
        row = self.connection.execute(
            """
            SELECT prompt, seq_len, hidden_dim, dtype, data
            FROM features WHERE cache_key=?
            """,
            (text_cache_key(prompt),),
        ).fetchone()
        if row is None:
            raise KeyError(
                "Prompt is missing from the text feature cache. "
                f"key={text_cache_key(prompt)}, prompt={prompt[:120]!r}"
            )
        stored_prompt, seq_len, hidden_dim, dtype_name, blob = row
        if stored_prompt != prompt:
            raise RuntimeError("SHA-256 collision or corrupted text cache entry.")
        return _blob_to_tensor(
            blob, dtype_name, (int(seq_len), int(hidden_dim))
        )

    def get_batch(
        self,
        texts: Iterable[str],
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [self.get(text) for text in texts]
        if not rows:
            raise ValueError("Cannot build an empty cached-text batch.")
        hidden_dims = {int(row.shape[1]) for row in rows}
        dtypes = {row.dtype for row in rows}
        if len(hidden_dims) != 1 or len(dtypes) != 1:
            raise ValueError("Inconsistent hidden dimensions or dtypes in cache.")
        max_len = max(int(row.shape[0]) for row in rows)
        hidden_dim = hidden_dims.pop()
        dtype = dtypes.pop()
        features = torch.zeros(
            len(rows), max_len, hidden_dim, dtype=dtype
        )
        attention_mask = torch.zeros(len(rows), max_len, dtype=torch.long)
        for index, row in enumerate(rows):
            seq_len = int(row.shape[0])
            features[index, -seq_len:] = row
            attention_mask[index, -seq_len:] = 1
        if device is not None:
            features = features.to(device)
            attention_mask = attention_mask.to(device)
        return features, attention_mask

    def __len__(self) -> int:
        return int(
            self.connection.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        )

    def close(self) -> None:
        self.connection.close()
