#!/usr/bin/env python3
"""Import Cursor Agent chats into Codex CLI resume state."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse


SESSION_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "cursor-to-codex-resume")
ENV_CHAT_ID_KEYS = (
    "CURSOR_CHAT_ID",
    "CURSOR_AGENT_ID",
    "CURSOR_SESSION_ID",
    "CURSOR_CONVERSATION_ID",
    "CURSOR_THREAD_ID",
    "AGENT_CHAT_ID",
    "AGENT_SESSION_ID",
)
ENV_TRANSCRIPT_KEYS = (
    "CURSOR_TRANSCRIPT_PATH",
    "CURSOR_AGENT_TRANSCRIPT",
    "CURSOR_AGENT_TRANSCRIPT_PATH",
)
HANDOFF_COMMAND_NAMES = (
    "cursor-to-codex-resume",
    "/cursor-to-codex-resume",
    "$cursor-to-codex-resume",
)
EXPORT_SCHEMA_VERSION = 24
DEFAULT_MODEL_CONTEXT_WINDOW = 258400
DEFAULT_REPLAY_MAX_CHARS = 900
DEFAULT_REPLAY_MAX_LINES = 8
DEFAULT_REPLAY_MAX_LINE_CHARS = 120
DEFAULT_REPLAY_SPLIT_MAX_CHARS = 900
DEFAULT_REPLAY_SPLIT_MAX_LINES = 10
DEFAULT_TOOL_OUTPUT_PREVIEW_LINES = 5
DEFAULT_TOOL_COMMAND_PREVIEW_LINES = 6
DEFAULT_PATCH_PREVIEW_LINES = 16
DEFAULT_EXEC_MAX_OUTPUT_TOKENS = 20000
EXEC_TOOL_NAMES = {"shell", "read", "grep", "glob", "webfetch"}
PATCH_TOOL_NAMES = {"write", "strreplace", "edit", "delete"}
REPLAY_MODES = {"split", "compact", "full"}
TOOL_REPLAY_MODES = {"compact", "none"}
DEFAULT_TOOL_REPLAY_MODE = "none"
PATCH_FAILURE_MARKERS = (
    "old_string not found",
    "could not find",
    "couldn't find",
    "not found",
    "no match",
    "did not apply",
    "not applied",
    "failed",
    "failure",
    "error",
    "unable to",
)
PATCH_SUCCESS_MARKERS = (
    "ok",
    "done",
    "success",
    "succeeded",
    "successfully",
    "applied",
    "updated",
    "created",
    "deleted",
    "written",
)


@dataclass
class Candidate:
    chat_id: str
    title: str | None = None
    created_ms: int | None = None
    updated_ms: int | None = None
    transcript_path: Path | None = None
    chat_dir: Path | None = None
    store_db: Path | None = None
    workspace_uri: str | None = None
    project_key: str | None = None


@dataclass
class ExportEntry:
    kind: str
    text: str | None = None
    phase: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    args: Any = None
    result: Any = None


@dataclass
class ThreadDefaults:
    cli_version: str
    model: str | None = None
    reasoning_effort: str | None = None
    approval_mode: str = "on-request"


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def maybe_resolve(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser().absolute()


def utc_iso(ms: int) -> str:
    return (
        datetime.fromtimestamp(ms / 1000, timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def local_rollout_parts(ms: int, session_id: str) -> tuple[str, str, str, str]:
    dt = datetime.fromtimestamp(ms / 1000).astimezone()
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    day = dt.strftime("%d")
    stamp = dt.strftime("%Y-%m-%dT%H-%M-%S")
    return year, month, day, f"rollout-{stamp}-{session_id}.jsonl"


def uuid7_like() -> str:
    ms = int(time.time() * 1000)
    rand = os.urandom(10)
    data = bytearray(16)
    for index in range(6):
        data[5 - index] = (ms >> (8 * index)) & 0xFF
    data[6] = (rand[0] & 0x0F) | 0x70
    data[7] = rand[1]
    data[8] = (rand[2] & 0x3F) | 0x80
    data[9:16] = rand[3:10]
    return str(uuid.UUID(bytes=bytes(data)))


def item_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def turn_metadata(turn_id: str | None) -> dict[str, Any]:
    if not turn_id:
        return {}
    return {"internal_chat_message_metadata_passthrough": {"turn_id": turn_id}}


def stable_call_id(source_id: str | None, tool_name: str | None, index: int) -> str:
    seed = f"{source_id or ''}:{tool_name or ''}:{index}"
    return "call_" + uuid.uuid5(SESSION_NAMESPACE, seed).hex[:24]


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def clamp_replay_lines(text: str, max_line_chars: int = DEFAULT_REPLAY_MAX_LINE_CHARS) -> str:
    lines = []
    for line in text.splitlines():
        if len(line) <= max_line_chars:
            lines.append(line)
        else:
            lines.append(line[:max_line_chars].rstrip() + "...")
    return "\n".join(lines)


def compact_replay_text(
    text: str,
    *,
    enabled: bool,
    max_chars: int = DEFAULT_REPLAY_MAX_CHARS,
    max_lines: int = DEFAULT_REPLAY_MAX_LINES,
) -> str:
    if not enabled:
        return text
    if len(text) <= max_chars and len(text.splitlines()) <= max_lines:
        return clamp_replay_lines(text)

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    if paragraphs and len(paragraphs[0]) <= max_chars and len(paragraphs[0].splitlines()) <= max_lines:
        return clamp_replay_lines(paragraphs[0].rstrip()) + "\n..."

    lines = text.strip().splitlines()
    clipped = "\n".join(lines[:max_lines]).strip()
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars].rstrip()
    return clamp_replay_lines(clipped) + "\n..."


def split_replay_text(
    text: str,
    *,
    max_chars: int = DEFAULT_REPLAY_SPLIT_MAX_CHARS,
    max_lines: int = DEFAULT_REPLAY_SPLIT_MAX_LINES,
) -> list[str]:
    if not text:
        return [""]
    if len(text) <= max_chars and len(text.splitlines()) <= max_lines:
        return [text]

    blocks: list[str] = []
    block: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush_block() -> None:
        nonlocal block
        if block:
            blocks.append("".join(block))
            block = []

    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        fence_match = re.match(r"(```+|~~~+)", stripped)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                flush_block()
                in_fence = True
                fence_marker = marker[:3]
                block.append(line)
                continue
            if marker.startswith(fence_marker):
                block.append(line)
                in_fence = False
                fence_marker = ""
                flush_block()
                continue

        block.append(line)
        if not in_fence and not line.strip():
            flush_block()
    flush_block()

    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    current_lines = 0

    def flush() -> None:
        nonlocal current, current_chars, current_lines
        if current:
            chunks.append("".join(current).rstrip("\n"))
            current = []
            current_chars = 0
            current_lines = 0

    def add_piece(piece: str) -> None:
        nonlocal current_chars, current_lines
        piece_lines = len(piece.splitlines()) or 1
        if current and (
            current_chars + len(piece) > max_chars
            or current_lines + piece_lines > max_lines
        ):
            flush()
        current.append(piece)
        current_chars += len(piece)
        current_lines += piece_lines

    for piece in blocks:
        piece_is_fence = piece.lstrip().startswith(("```", "~~~"))
        if (
            piece_is_fence
            or (len(piece) <= max_chars and len(piece.splitlines()) <= max_lines)
        ):
            add_piece(piece)
            continue

        flush()
        line_chunk: list[str] = []
        line_chars = 0
        for line in piece.splitlines(keepends=True):
            if line_chunk and (
                line_chars + len(line) > max_chars
                or len(line_chunk) >= max_lines
            ):
                chunks.append("".join(line_chunk).rstrip("\n"))
                line_chunk = []
                line_chars = 0
            if len(line) > max_chars:
                if line_chunk:
                    chunks.append("".join(line_chunk).rstrip("\n"))
                    line_chunk = []
                    line_chars = 0
                for start in range(0, len(line), max_chars):
                    chunks.append(line[start : start + max_chars].rstrip("\n"))
                continue
            line_chunk.append(line)
            line_chars += len(line)
        if line_chunk:
            chunks.append("".join(line_chunk).rstrip("\n"))
    flush()
    return [chunk for chunk in chunks if chunk or text == ""]


def replay_texts(text: str, mode: str) -> list[str]:
    if mode == "full":
        return [text]
    text = render_markdown_tables_for_replay(text)
    if mode == "compact":
        return [compact_replay_text(text, enabled=True)]
    if mode == "split":
        return split_replay_text(text)
    raise ValueError(f"unknown replay mode: {mode}")


def visible_width(text: str) -> int:
    import unicodedata

    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def pad_visible(text: str, width: int) -> str:
    return text + " " * max(0, width - visible_width(text))


def parse_markdown_table_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if "|" not in stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    return cells if len(cells) >= 2 else None


def is_markdown_table_separator(line: str) -> bool:
    cells = parse_markdown_table_row(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def render_plain_table(rows: list[list[str]]) -> str:
    widths = [0] * max(len(row) for row in rows)
    normalized = [
        [plain_table_cell(cell) for cell in row] + [""] * (len(widths) - len(row))
        for row in rows
    ]
    for row in normalized:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], visible_width(cell))

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    rendered = [border("┌", "┬", "┐")]
    for index, row in enumerate(normalized):
        rendered.append("│ " + " │ ".join(pad_visible(cell, widths[i]) for i, cell in enumerate(row)) + " │")
        if index == 0 and len(normalized) > 1:
            rendered.append(border("├", "┼", "┤"))
    rendered.append(border("└", "┴", "┘"))
    return "\n".join(rendered)


def plain_table_cell(cell: str) -> str:
    cell = re.sub(r"`([^`]*)`", r"\1", cell)
    cell = re.sub(r"\*\*([^*]*)\*\*", r"\1", cell)
    cell = re.sub(r"\*([^*]*)\*", r"\1", cell)
    return cell


def render_markdown_tables_for_replay(text: str) -> str:
    lines = text.splitlines()
    rendered: list[str] = []
    index = 0
    in_fence = False
    fence_marker = ""
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        fence_match = re.match(r"(```+|~~~+)", stripped)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker[:3]
            elif marker.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            rendered.append(line)
            index += 1
            continue

        if not in_fence and index + 1 < len(lines):
            header = parse_markdown_table_row(lines[index])
            if header and is_markdown_table_separator(lines[index + 1]):
                table_rows = [header]
                index += 2
                while index < len(lines):
                    row = parse_markdown_table_row(lines[index])
                    if not row:
                        break
                    table_rows.append(row)
                    index += 1
                rendered.append(render_plain_table(table_rows))
                continue

        rendered.append(line)
        index += 1

    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(rendered) + suffix


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_varint(data: bytes, index: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while index < len(data):
        byte = data[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7
    raise ValueError("truncated protobuf varint")


def iter_proto_fields(data: bytes) -> Iterable[tuple[int, int, Any]]:
    index = 0
    while index < len(data):
        tag, index = read_varint(data, index)
        field = tag >> 3
        wire = tag & 7
        if wire == 0:
            value, index = read_varint(data, index)
            yield field, wire, value
        elif wire == 1:
            value = data[index : index + 8]
            index += 8
            yield field, wire, value
        elif wire == 2:
            length, index = read_varint(data, index)
            value = data[index : index + length]
            index += length
            yield field, wire, value
        elif wire == 5:
            value = data[index : index + 4]
            index += 4
            yield field, wire, value
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def read_cursor_store_meta(store_db: Path) -> dict[str, Any]:
    if not store_db.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{store_db}?mode=ro", uri=True)
        row = con.execute("select value from meta where key = '0'").fetchone()
        con.close()
    except sqlite3.Error:
        return {}
    if not row:
        return {}
    value = row[0]
    try:
        return json.loads(bytes.fromhex(value).decode("utf-8"))
    except Exception:
        return {}


def read_root_blob(store_db: Path, root_blob_id: str) -> bytes | None:
    try:
        con = sqlite3.connect(f"file:{store_db}?mode=ro", uri=True)
        row = con.execute("select data from blobs where id = ?", (root_blob_id,)).fetchone()
        con.close()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def root_blob_ids_and_workspace(store_db: Path, root_blob_id: str) -> tuple[list[str], str | None]:
    data = read_root_blob(store_db, root_blob_id)
    if not data:
        return [], None
    blob_ids: list[str] = []
    workspace_uri = None
    try:
        for field, wire, value in iter_proto_fields(data):
            if field == 1 and wire == 2 and isinstance(value, bytes) and len(value) == 32:
                blob_ids.append(value.hex())
            elif field == 9 and wire == 2 and isinstance(value, bytes):
                try:
                    workspace_uri = value.decode("utf-8")
                except UnicodeDecodeError:
                    pass
    except ValueError:
        pass
    return blob_ids, workspace_uri


def merge_candidate(target: dict[str, Candidate], incoming: Candidate) -> None:
    existing = target.get(incoming.chat_id)
    if existing is None:
        target[incoming.chat_id] = incoming
        return
    for field in (
        "title",
        "created_ms",
        "updated_ms",
        "transcript_path",
        "chat_dir",
        "store_db",
        "workspace_uri",
        "project_key",
    ):
        if getattr(existing, field) in (None, "") and getattr(incoming, field) not in (None, ""):
            setattr(existing, field, getattr(incoming, field))
    if incoming.updated_ms and (not existing.updated_ms or incoming.updated_ms > existing.updated_ms):
        existing.updated_ms = incoming.updated_ms
    if incoming.created_ms and (not existing.created_ms or incoming.created_ms < existing.created_ms):
        existing.created_ms = incoming.created_ms


def discover_candidates(cursor_home: Path) -> list[Candidate]:
    candidates: dict[str, Candidate] = {}

    chats_root = cursor_home / "chats"
    if chats_root.exists():
        for meta_path in chats_root.glob("*/*/meta.json"):
            chat_dir = meta_path.parent
            meta = read_json(meta_path)
            chat_id = chat_dir.name
            store_db = chat_dir / "store.db"
            store_meta = read_cursor_store_meta(store_db)
            workspace_uri = None
            root_blob_id = store_meta.get("latestRootBlobId")
            if isinstance(root_blob_id, str):
                _, workspace_uri = root_blob_ids_and_workspace(store_db, root_blob_id)
            merge_candidate(
                candidates,
                Candidate(
                    chat_id=chat_id,
                    title=meta.get("title") or store_meta.get("name"),
                    created_ms=meta.get("createdAtMs") or store_meta.get("createdAt"),
                    updated_ms=meta.get("updatedAtMs") or int(meta_path.stat().st_mtime * 1000),
                    chat_dir=chat_dir,
                    store_db=store_db if store_db.exists() else None,
                    workspace_uri=workspace_uri,
                ),
            )

    projects_root = cursor_home / "projects"
    if projects_root.exists():
        for transcript in projects_root.glob("*/agent-transcripts/*/*.jsonl"):
            chat_id = transcript.parent.name
            project_key = transcript.parents[2].name
            merge_candidate(
                candidates,
                Candidate(
                    chat_id=chat_id,
                    updated_ms=int(transcript.stat().st_mtime * 1000),
                    transcript_path=transcript,
                    project_key=project_key,
                ),
            )

    return sorted(candidates.values(), key=lambda c: c.updated_ms or 0, reverse=True)


def project_key_to_path(project_key: str | None) -> Path | None:
    if not project_key:
        return None
    parts = project_key.split("-")
    if len(parts) < 2:
        return None
    roots = []
    if parts[0] == "home" and len(parts) >= 2:
        roots.append(Path("/home") / parts[1])
    if parts[0] == "tmp":
        roots.append(Path("/tmp"))
    for root in roots:
        if root.exists() and len(parts) == 2:
            return root
        suffix = "-".join(parts[2:]) if parts[0] == "home" else "-".join(parts[1:])
        if suffix:
            direct = root / suffix
            if direct.exists():
                return direct
            underscore = root / suffix.replace("-", "_")
            if underscore.exists():
                return underscore
    return None


def cursor_project_key_for_path(path: Path) -> str:
    raw = str(maybe_resolve(path)).strip("/")
    chars = []
    previous_dash = False
    for char in raw:
        if char.isalnum():
            chars.append(char.lower())
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    return "".join(chars).strip("-")


def cursor_project_keys_for_cwd(cwd: Path) -> set[str]:
    return set(cursor_project_key_chain_for_cwd(cwd))


def cursor_project_key_chain_for_cwd(cwd: Path) -> list[str]:
    resolved = maybe_resolve(cwd)
    keys = []
    seen = set()
    for path in (resolved, *resolved.parents):
        key = cursor_project_key_for_path(path)
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def workspace_uri_to_path(uri: str | None) -> Path | None:
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
        return path if path.exists() else None
    path = Path(uri)
    return path if path.exists() else None


def candidate_cwd(candidate: Candidate, fallback: Path) -> Path:
    return workspace_uri_to_path(candidate.workspace_uri) or project_key_to_path(candidate.project_key) or fallback


def candidate_workspace_path(candidate: Candidate) -> Path | None:
    return workspace_uri_to_path(candidate.workspace_uri) or project_key_to_path(candidate.project_key)


def paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = maybe_resolve(left)
    right_resolved = maybe_resolve(right)
    if left_resolved == right_resolved:
        return True
    try:
        left_resolved.relative_to(right_resolved)
        return True
    except ValueError:
        pass
    try:
        right_resolved.relative_to(left_resolved)
        return True
    except ValueError:
        return False


def workspace_match_rank(cwd: Path, workspace: Path) -> tuple[int, int] | None:
    cwd_resolved = maybe_resolve(cwd)
    workspace_resolved = maybe_resolve(workspace)
    if cwd_resolved == workspace_resolved:
        return (0, 0)
    try:
        relative = cwd_resolved.relative_to(workspace_resolved)
        return (1, len(relative.parts))
    except ValueError:
        pass
    try:
        relative = workspace_resolved.relative_to(cwd_resolved)
        return (2, len(relative.parts))
    except ValueError:
        return None


def id_matches(candidate: Candidate, value: str) -> bool:
    normalized = value.strip().casefold()
    if not normalized:
        return False
    candidate_id = candidate.chat_id.casefold()
    return candidate_id == normalized or candidate_id.startswith(normalized) or normalized.startswith(candidate_id)


def select_current_candidate(
    usable: list[Candidate],
    cwd: Path,
) -> tuple[Candidate, str]:
    for key in ENV_TRANSCRIPT_KEYS:
        value = os.environ.get(key)
        if not value:
            continue
        transcript = maybe_resolve(Path(value))
        for candidate in usable:
            if candidate.transcript_path and maybe_resolve(candidate.transcript_path) == transcript:
                return candidate, f"env:{key}"

    for key in ENV_CHAT_ID_KEYS:
        value = os.environ.get(key)
        if not value:
            continue
        matches = [candidate for candidate in usable if id_matches(candidate, value)]
        if len(matches) == 1:
            return matches[0], f"env:{key}"

    current_project_keys = cursor_project_key_chain_for_cwd(cwd)
    project_key_rank = {key: index for index, key in enumerate(current_project_keys)}
    project_matches = [
        candidate
        for candidate in usable
        if candidate.project_key and candidate.project_key in project_key_rank
    ]
    if project_matches:
        selected = sorted(
            project_matches,
            key=lambda c: (project_key_rank[str(c.project_key)], -(c.updated_ms or 0)),
        )[0]
        return selected, "current-project"

    path_matches = []
    for candidate in usable:
        workspace = candidate_workspace_path(candidate)
        if workspace:
            rank = workspace_match_rank(cwd, workspace)
            if rank is not None:
                path_matches.append((rank, candidate))
    if path_matches:
        selected = sorted(
            path_matches,
            key=lambda item: (item[0][0], item[0][1], -(item[1].updated_ms or 0)),
        )[0][1]
        return selected, "current-workspace"

    raise SystemExit(
        "Could not identify the active Cursor session from the current workspace. "
        "Run with --list and pass --chat <id> to import a specific session."
    )


def select_candidate(
    candidates: list[Candidate],
    selector: str | None,
    current_cwd: Path,
) -> tuple[Candidate, str]:
    usable = [c for c in candidates if c.transcript_path or c.store_db]
    if not usable:
        raise SystemExit("No Cursor chats with transcript or store.db were found.")
    if not selector or selector == "current":
        return select_current_candidate(usable, current_cwd)
    if selector == "latest":
        return usable[0], "latest"

    maybe_path = Path(selector).expanduser()
    if maybe_path.exists():
        resolved = maybe_path.resolve()
        for candidate in usable:
            if resolved in (candidate.transcript_path, candidate.chat_dir, candidate.store_db):
                return candidate, "path"
        if resolved.is_dir() and (resolved / "store.db").exists():
            chat_id = resolved.name
            meta = read_json(resolved / "meta.json")
            return (
                Candidate(
                    chat_id=chat_id,
                    title=meta.get("title"),
                    created_ms=meta.get("createdAtMs"),
                    updated_ms=meta.get("updatedAtMs") or int(resolved.stat().st_mtime * 1000),
                    chat_dir=resolved,
                    store_db=resolved / "store.db",
                ),
                "path",
            )
        if resolved.is_file() and resolved.suffix == ".jsonl":
            chat_id = resolved.stem
            return (
                Candidate(
                    chat_id=chat_id,
                    updated_ms=int(resolved.stat().st_mtime * 1000),
                    transcript_path=resolved,
                ),
                "path",
            )

    query = selector.casefold()
    matches = [
        c
        for c in usable
        if c.chat_id.casefold().startswith(query)
        or (c.title and query in c.title.casefold())
    ]
    if len(matches) == 1:
        return matches[0], "query"
    if len(matches) > 1:
        lines = "\n".join(format_candidate(i + 1, c) for i, c in enumerate(matches[:20]))
        raise SystemExit(f"More than one Cursor chat matched {selector!r}:\n{lines}")
    raise SystemExit(f"No Cursor chat matched {selector!r}. Run with --list.")


def format_candidate(index: int, candidate: Candidate) -> str:
    updated = utc_iso(candidate.updated_ms) if candidate.updated_ms else "unknown"
    title = candidate.title or "(untitled)"
    source = candidate_source(candidate)
    project = candidate.project_key or ""
    return f"{index:>3}  {updated}  {candidate.chat_id}  {source:10}  {project:28}  {title}"


def candidate_source(candidate: Candidate) -> str:
    if candidate.transcript_path:
        return "transcript"
    if candidate.store_db:
        return "store.db"
    return "metadata"


def candidate_updated_at(candidate: Candidate) -> str | None:
    return utc_iso(candidate.updated_ms) if candidate.updated_ms else None


def candidate_debug_summary(candidate: Candidate) -> dict[str, Any]:
    return {
        "chat_id": candidate.chat_id,
        "title": candidate.title,
        "updated_at": candidate_updated_at(candidate),
        "source": candidate_source(candidate),
        "project_key": candidate.project_key,
        "workspace_uri": candidate.workspace_uri,
        "transcript_path": str(candidate.transcript_path) if candidate.transcript_path else None,
        "store_db": str(candidate.store_db) if candidate.store_db else None,
    }


def strip_tagged_blocks(text: str, tag: str) -> str:
    pattern = re.compile(rf"<{re.escape(tag)}>\s*.*?\s*</{re.escape(tag)}>", re.DOTALL)
    return pattern.sub("", text)


def unwrap_tag(text: str, tag: str) -> str:
    stripped = text.strip()
    start = f"<{tag}>"
    end = f"</{tag}>"
    if stripped.startswith(start) and stripped.endswith(end):
        return stripped[len(start) : -len(end)].strip()
    return stripped


def normalize_export_text(role: str, text: str) -> str:
    text = text.strip()
    if role == "user":
        text = strip_tagged_blocks(text, "manually_attached_skills")
        text = unwrap_tag(text, "user_query")
    if role == "assistant":
        lines = [line for line in text.splitlines() if line.strip() != "[REDACTED]"]
        text = "\n".join(lines)
    return text.strip()


def is_handoff_invocation(text: str) -> bool:
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not first_line:
        return False
    first_word = first_line.split(maxsplit=1)[0]
    return first_word in HANDOFF_COMMAND_NAMES


def is_cursor_environment_text(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("<user_info>") or stripped.startswith("<rules>")


def clean_entries_for_export(entries: list[ExportEntry]) -> list[ExportEntry]:
    cleaned: list[ExportEntry] = []
    for entry in entries:
        if entry.kind not in {"user", "assistant"}:
            cleaned.append(entry)
            continue
        role = entry.kind
        text = normalize_export_text(role, entry.text or "")
        if not text:
            continue
        if role == "user" and is_cursor_environment_text(text):
            continue
        if role == "user" and is_handoff_invocation(text):
            break
        cleaned.append(
            ExportEntry(
                kind=entry.kind,
                text=text,
                phase=entry.phase,
                tool_name=entry.tool_name,
                tool_call_id=entry.tool_call_id,
                args=entry.args,
                result=entry.result,
            )
        )
    return cleaned


def text_from_content(content: Any, include_tools: bool) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"].strip()
        return ""
    if not isinstance(content, list):
        return ""

    chunks: list[str] = []
    for part in content:
        if isinstance(part, str):
            chunks.append(part)
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text" and isinstance(part.get("text"), str):
            chunks.append(part["text"])
        elif include_tools and part_type in {"tool_use", "tool-call"}:
            name = part.get("name") or part.get("toolName") or "tool"
            payload = part.get("input") if "input" in part else part.get("args")
            chunks.append(
                "[Cursor tool call: "
                + str(name)
                + "]\n"
                + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            )
        elif include_tools and part_type in {"tool_result", "tool-result"}:
            name = part.get("name") or part.get("toolName") or "tool"
            result = part.get("result", part.get("content", ""))
            chunks.append(f"[Cursor tool result: {name}]\n{result}")
    return "\n\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip()).strip()


def extract_entries_from_transcript(path: Path, include_tools: bool) -> list[ExportEntry]:
    entries: list[ExportEntry] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = item.get("role")
            if role not in {"user", "assistant", "tool"}:
                continue
            message = item.get("message") or {}
            if role == "tool":
                if not include_tools:
                    continue
                for part in iter_content_parts(message.get("content")):
                    if part.get("type") not in {"tool-result", "tool_result"}:
                        continue
                    entries.append(
                        ExportEntry(
                            kind="tool_result",
                            tool_name=part.get("toolName") or part.get("name"),
                            tool_call_id=part.get("toolCallId") or part.get("id") or item.get("id"),
                            result=extract_tool_result_text(part),
                        )
                    )
                continue
            if role == "user":
                text = text_from_content(message.get("content"), include_tools=False)
                if text:
                    entries.append(ExportEntry(kind="user", text=text))
                continue

            parts = list(iter_content_parts(message.get("content")))
            has_tool_call = any(part.get("type") in {"tool-call", "tool_use"} for part in parts)
            for part in parts:
                part_type = part.get("type")
                if part_type == "text" and isinstance(part.get("text"), str):
                    text = normalize_export_text("assistant", part["text"])
                    if text:
                        entries.append(
                            ExportEntry(
                                kind="assistant",
                                text=text,
                                phase="commentary" if has_tool_call else "final_answer",
                            )
                        )
                elif include_tools and part_type in {"tool-call", "tool_use"}:
                    entries.append(
                        ExportEntry(
                            kind="tool_call",
                            tool_name=part.get("toolName") or part.get("name"),
                            tool_call_id=part.get("toolCallId") or part.get("id"),
                            args=part.get("args") if "args" in part else part.get("input"),
                        )
                    )
                elif include_tools and part_type in {"tool-result", "tool_result"}:
                    entries.append(
                        ExportEntry(
                            kind="tool_result",
                            tool_name=part.get("toolName") or part.get("name"),
                            tool_call_id=part.get("toolCallId") or part.get("id") or item.get("id"),
                            result=extract_tool_result_text(part),
                        )
                    )
    return entries


def iter_content_parts(content: Any) -> Iterable[dict[str, Any]]:
    if isinstance(content, str):
        yield {"type": "text", "text": content}
    elif isinstance(content, dict):
        yield content
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                yield {"type": "text", "text": part}
            elif isinstance(part, dict):
                yield part


def extract_tool_result_text(part: dict[str, Any]) -> str:
    result = part.get("result")
    if isinstance(result, str):
        return result
    if result is not None:
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    experimental = part.get("experimental_content")
    if isinstance(experimental, list):
        chunks = []
        for item in experimental:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        if chunks:
            return "\n\n".join(chunks)
    return ""


def extract_entries_from_store(store_db: Path, include_tools: bool) -> tuple[list[ExportEntry], str | None]:
    store_meta = read_cursor_store_meta(store_db)
    root_blob_id = store_meta.get("latestRootBlobId")
    if not isinstance(root_blob_id, str):
        return [], None
    blob_ids, workspace_uri = root_blob_ids_and_workspace(store_db, root_blob_id)
    entries: list[ExportEntry] = []
    stopped = False
    try:
        con = sqlite3.connect(f"file:{store_db}?mode=ro", uri=True)
        for blob_id in blob_ids:
            row = con.execute("select data from blobs where id = ?", (blob_id,)).fetchone()
            if not row:
                continue
            try:
                item = json.loads(row[0].decode("utf-8"))
            except Exception:
                continue
            role = item.get("role")
            if role == "system":
                continue
            if role == "user":
                text = text_from_content(item.get("content"), include_tools=False)
                text = normalize_export_text("user", text)
                if not text or is_cursor_environment_text(text):
                    continue
                if is_handoff_invocation(text):
                    stopped = True
                    break
                entries.append(ExportEntry(kind="user", text=text))
                continue
            if role == "assistant":
                parts = list(iter_content_parts(item.get("content")))
                has_tool_call = any(part.get("type") in {"tool-call", "tool_use"} for part in parts)
                for part in parts:
                    part_type = part.get("type")
                    if part_type in {"redacted-reasoning", "reasoning"}:
                        continue
                    elif part_type == "text" and isinstance(part.get("text"), str):
                        text = normalize_export_text("assistant", part["text"])
                        if text:
                            entries.append(
                                ExportEntry(
                                    kind="assistant",
                                    text=text,
                                    phase="commentary" if has_tool_call else "final_answer",
                                )
                            )
                    elif include_tools and part_type in {"tool-call", "tool_use"}:
                        entries.append(
                            ExportEntry(
                                kind="tool_call",
                                tool_name=part.get("toolName") or part.get("name"),
                                tool_call_id=part.get("toolCallId") or part.get("id"),
                                args=part.get("args") if "args" in part else part.get("input"),
                            )
                        )
                continue
            if role == "tool" and include_tools:
                for part in iter_content_parts(item.get("content")):
                    if part.get("type") not in {"tool-result", "tool_result"}:
                        continue
                    entries.append(
                        ExportEntry(
                            kind="tool_result",
                            tool_name=part.get("toolName") or part.get("name"),
                            tool_call_id=part.get("toolCallId") or part.get("id") or item.get("id"),
                            result=extract_tool_result_text(part),
                        )
                    )
        con.close()
    except sqlite3.Error:
        return entries, workspace_uri
    if stopped:
        return entries, workspace_uri
    return entries, workspace_uri


def extract_entries(candidate: Candidate, include_tools: bool) -> list[ExportEntry]:
    if candidate.store_db and candidate.store_db.exists():
        entries, workspace_uri = extract_entries_from_store(candidate.store_db, include_tools)
        if workspace_uri and not candidate.workspace_uri:
            candidate.workspace_uri = workspace_uri
        cleaned = clean_entries_for_export(entries)
        if cleaned:
            return cleaned
    if candidate.transcript_path and candidate.transcript_path.exists():
        entries = extract_entries_from_transcript(candidate.transcript_path, include_tools)
        if entries:
            return clean_entries_for_export(entries)
    return []


def session_id_for(candidate: Candidate, new: bool) -> str:
    if new:
        return str(uuid.uuid4())
    return str(uuid.uuid5(SESSION_NAMESPACE, candidate.chat_id))


def rollout_needs_rewrite(
    path: Path,
    expected_replay: str = "split",
    expected_tool_replay: str = DEFAULT_TOOL_REPLAY_MODE,
) -> bool:
    if not path.exists():
        return False
    saw_user_message = False
    saw_user_event = False
    saw_assistant_message = False
    saw_agent_message = False
    saw_task_started = False
    saw_turn_context = False
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    return True
                payload = item.get("payload")
                if not isinstance(payload, dict):
                    continue
                if item.get("type") == "session_meta":
                    if payload.get("cli_version") == "cursor-to-codex-resume":
                        return True
                    version = payload.get("cursor_to_codex_resume_version")
                    if version != EXPORT_SCHEMA_VERSION:
                        return True
                    if payload.get("cursor_to_codex_resume_replay") != expected_replay:
                        return True
                    if payload.get("cursor_to_codex_resume_tool_replay", DEFAULT_TOOL_REPLAY_MODE) != expected_tool_replay:
                        return True
                if payload.get("metadata") is not None:
                    return True
                if payload.get("type") == "reasoning" and payload.get("encrypted_content"):
                    return True
                if item.get("type") == "event_msg" and payload.get("type") in {"user_message", "agent_message"}:
                    message = payload.get("message")
                    if (
                        isinstance(message, str)
                        and (
                            len(message) > DEFAULT_REPLAY_SPLIT_MAX_CHARS
                            or len(message.splitlines()) > DEFAULT_REPLAY_SPLIT_MAX_LINES
                        )
                    ):
                        return True
                if item.get("type") == "turn_context":
                    saw_turn_context = True
                if item.get("type") == "event_msg" and payload.get("type") == "task_started":
                    saw_task_started = True
                if item.get("type") == "event_msg" and payload.get("type") == "user_message":
                    saw_user_event = True
                if item.get("type") == "event_msg" and payload.get("type") == "agent_message":
                    saw_agent_message = True
                if payload.get("type") == "message" and payload.get("role") == "assistant":
                    saw_assistant_message = True
                    if payload.get("phase") == "final":
                        return True
                if payload.get("type") == "message" and payload.get("role") == "user":
                    saw_user_message = True
                    for content in payload.get("content", []):
                        if not isinstance(content, dict):
                            continue
                        text = content.get("text")
                        if not isinstance(text, str):
                            continue
                        normalized = normalize_export_text("user", text)
                        if normalized != text.strip() or is_handoff_invocation(normalized):
                            return True
    except OSError:
        return False
    if saw_user_message and not saw_user_event:
        return True
    if saw_assistant_message and not saw_agent_message:
        return True
    if saw_user_message and (not saw_task_started or not saw_turn_context):
        return True
    return False


def turn_context_payload(
    *,
    turn_id: str,
    cwd: Path,
    created_ms: int,
    defaults: ThreadDefaults,
) -> dict[str, Any]:
    local_dt = datetime.fromtimestamp(created_ms / 1000).astimezone()
    return {
        "turn_id": turn_id,
        "cwd": str(cwd),
        "workspace_roots": [str(cwd)],
        "current_date": local_dt.strftime("%Y-%m-%d"),
        "timezone": local_dt.tzname() or "local",
        "approval_policy": defaults.approval_mode,
        "sandbox_policy": {"type": "disabled"},
        "file_system_sandbox_policy": {"type": "disabled"},
        "permission_profile": {"type": "disabled"},
        "model": defaults.model,
        "effort": defaults.reasoning_effort,
        "collaboration_mode": "default",
        "realtime_active": False,
    }


def token_count_event(timestamp_ms: int, input_tokens: int, output_tokens: int) -> dict[str, Any]:
    total = input_tokens + output_tokens
    return {
        "timestamp": utc_iso(timestamp_ms),
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": max(0, input_tokens // 3),
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": 0,
                    "total_tokens": total,
                },
                "last_token_usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": max(0, input_tokens // 3),
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": 0,
                    "total_tokens": total,
                },
                "model_context_window": DEFAULT_MODEL_CONTEXT_WINDOW,
            },
            "rate_limits": {
                "limit_id": "codex",
                "limit_name": None,
                "primary": {
                    "used_percent": 0.0,
                    "window_minutes": 300,
                    "resets_at": timestamp_ms // 1000 + 3600,
                },
                "secondary": {
                    "used_percent": 0.0,
                    "window_minutes": 10080,
                    "resets_at": timestamp_ms // 1000 + 86400,
                },
                "credits": None,
                "plan_type": "pro",
            },
        },
    }


def shell_cmd_for_tool(tool_name: str | None, args: Any, cwd: Path) -> str:
    data = args if isinstance(args, dict) else {}
    name = (tool_name or "").casefold()
    if name == "shell":
        return str(data.get("command") or data.get("cmd") or "").strip()
    if name == "read":
        path = str(data.get("path") or "")
        limit = data.get("limit")
        offset = data.get("offset")
        if isinstance(limit, int) and limit > 0:
            start = int(offset or 0) + 1
            end = start + limit - 1
            return f"sed -n {shlex.quote(f'{start},{end}p')} {shlex.quote(path)}"
        return f"cat {shlex.quote(path)}"
    if name == "grep":
        pattern = str(data.get("pattern") or "")
        path = str(data.get("path") or cwd)
        glob = data.get("glob")
        head_limit = data.get("head_limit")
        cmd = f"rg -n {shlex.quote(pattern)} {shlex.quote(path)}"
        if glob:
            cmd += f" -g {shlex.quote(str(glob))}"
        if isinstance(head_limit, int) and head_limit > 0:
            cmd += f" | head -{head_limit}"
        return cmd
    if name == "glob":
        target = str(data.get("target_directory") or cwd)
        pattern = str(data.get("glob_pattern") or "*")
        return f"find {shlex.quote(target)} -path {shlex.quote(pattern)} -print"
    if name == "webfetch":
        url = str(data.get("url") or "")
        return f"curl -L {shlex.quote(url)}"
    return ""


def tool_workdir(tool_name: str | None, args: Any, cwd: Path) -> str:
    data = args if isinstance(args, dict) else {}
    for key in ("workdir", "cwd"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return str(cwd)


def exec_max_output_tokens(tool_name: str | None) -> int:
    name = (tool_name or "").casefold()
    if name in {"read", "grep", "glob"}:
        return 12000
    return DEFAULT_EXEC_MAX_OUTPUT_TOKENS


def exec_command_arguments(tool_name: str | None, args: Any, cwd: Path) -> dict[str, Any] | None:
    cmd = shell_cmd_for_tool(tool_name, args, cwd).strip()
    if not cmd:
        return None
    return {
        "cmd": cmd,
        "workdir": tool_workdir(tool_name, args, cwd),
        "max_output_tokens": exec_max_output_tokens(tool_name),
    }


def web_search_action(args: Any) -> dict[str, Any] | None:
    if not isinstance(args, dict):
        return None
    url = args.get("url")
    if isinstance(url, str) and url.strip():
        return {"type": "open_page", "url": url.strip()}
    query = args.get("query") or args.get("search_term") or args.get("searchTerm")
    if not isinstance(query, str) or not query.strip():
        return None
    query = query.strip()
    queries = args.get("queries")
    if not isinstance(queries, list) or not all(isinstance(item, str) for item in queries):
        queries = [query]
    return {"type": "search", "query": query, "queries": queries}


def semantic_search_arguments(args: Any) -> dict[str, Any] | None:
    if not isinstance(args, dict):
        return None
    query = args.get("query") or args.get("search_term") or args.get("searchTerm")
    if not isinstance(query, str) or not query.strip():
        return None
    payload: dict[str, Any] = {"query": query.strip()}
    limit = args.get("num_results") or args.get("limit") or args.get("top_k")
    if isinstance(limit, int) and limit > 0:
        payload["limit"] = limit
    directories = args.get("target_directories") or args.get("targetDirectories")
    if isinstance(directories, list) and all(isinstance(item, str) for item in directories):
        payload["target_directories"] = directories
    return payload


def read_lints_arguments(args: Any) -> dict[str, Any] | None:
    if not isinstance(args, dict):
        return None
    paths = args.get("paths")
    if isinstance(paths, list):
        normalized = [path for path in paths if isinstance(path, str) and path.strip()]
        if len(normalized) == 1:
            return {"file": normalized[0]}
        if normalized:
            return {"paths": normalized}
    path = args.get("path") or args.get("file")
    if isinstance(path, str) and path.strip():
        return {"file": path.strip()}
    return None


def wait_arguments(args: Any) -> dict[str, Any] | None:
    if not isinstance(args, dict):
        return None
    raw_id = args.get("task_id") or args.get("taskId") or args.get("id")
    ids = args.get("ids")
    if isinstance(ids, list):
        normalized_ids = [str(item) for item in ids if str(item).strip()]
    elif raw_id is not None and str(raw_id).strip():
        normalized_ids = [str(raw_id).strip()]
    else:
        normalized_ids = []
    timeout = args.get("timeout_ms") or args.get("block_until_ms") or args.get("blockUntilMs")
    payload: dict[str, Any] = {"ids": normalized_ids}
    if isinstance(timeout, int) and timeout > 0:
        payload["timeout_ms"] = timeout
    else:
        payload["timeout_ms"] = 30000
    pattern = args.get("pattern")
    if isinstance(pattern, str) and pattern.strip():
        payload["pattern"] = pattern.strip()
    return payload


def normalize_function_name(tool_name: str | None) -> str:
    raw = (tool_name or "tool").strip()
    name = re.sub(r"[^0-9A-Za-z_]+", "_", raw)
    name = re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower().strip("_")
    return name or "tool"


def update_plan_arguments(args: Any) -> dict[str, Any] | None:
    if not isinstance(args, dict):
        return None
    todos = args.get("todos")
    if not isinstance(todos, list):
        return None
    status_map = {
        "in_progress": "in_progress",
        "pending": "pending",
        "completed": "completed",
        "complete": "completed",
        "done": "completed",
    }
    plan = []
    for item in todos:
        if not isinstance(item, dict):
            continue
        step = item.get("content") or item.get("title") or item.get("step")
        if not isinstance(step, str) or not step.strip():
            continue
        status = status_map.get(str(item.get("status") or "pending"), "pending")
        plan.append({"step": step.strip(), "status": status})
    if not plan:
        return None
    payload: dict[str, Any] = {"plan": plan}
    explanation = args.get("explanation")
    if isinstance(explanation, str) and explanation.strip():
        payload["explanation"] = explanation.strip()
    return payload


def generic_tool_arguments(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return args
    return {"input": args}


def patch_lines_from_text(prefix: str, text: str) -> list[str]:
    lines = text.splitlines()
    if text.endswith("\n"):
        trailing = lines
    else:
        trailing = lines
    return [prefix + line for line in trailing]


def git_head_file_content(path: str, cwd: Path | None) -> str | None:
    if cwd is None:
        return None
    try:
        repo_proc = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None
    if repo_proc.returncode != 0:
        return None
    repo_root = Path(repo_proc.stdout.strip())
    try:
        absolute = Path(path)
        if not absolute.is_absolute():
            absolute = cwd / absolute
        rel_path = absolute.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return None
    try:
        show_proc = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"HEAD:{rel_path}"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None
    if show_proc.returncode != 0:
        return None
    return show_proc.stdout


def full_update_patch(path: str, old: str, new: str) -> tuple[str, dict[str, Any]]:
    lines = ["*** Begin Patch", f"*** Update File: {path}", "@@"]
    lines.extend(patch_lines_from_text("-", old))
    lines.extend(patch_lines_from_text("+", new))
    lines.append("*** End Patch")
    diff = "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        )
    )
    return "\n".join(lines) + "\n", {path: {"type": "update", "unified_diff": diff, "move_path": None}}


def patch_for_tool(tool_name: str | None, args: Any, cwd: Path | None = None) -> tuple[str | None, dict[str, Any] | None]:
    if not isinstance(args, dict):
        return None, None
    name = (tool_name or "").casefold()
    path = args.get("path")
    if not isinstance(path, str) or not path:
        return None, None
    if name == "delete":
        lines = ["*** Begin Patch", f"*** Delete File: {path}", "*** End Patch"]
        return "\n".join(lines) + "\n", {path: {"type": "delete", "content": ""}}
    if name == "write" and isinstance(args.get("contents"), str):
        contents = args["contents"]
        old_contents = git_head_file_content(path, cwd)
        if old_contents is not None:
            return full_update_patch(path, old_contents, contents)
        lines = ["*** Begin Patch", f"*** Add File: {path}"]
        lines.extend(patch_lines_from_text("+", contents))
        lines.append("*** End Patch")
        return "\n".join(lines) + "\n", {path: {"type": "add", "content": contents}}
    if name in {"strreplace", "edit"}:
        old = args.get("old_string") or args.get("oldString")
        new = args.get("new_string") or args.get("newString")
        if isinstance(old, str) and isinstance(new, str):
            return full_update_patch(path, old, new)
    return None, None


def display_path(path: str, cwd: Path) -> str:
    try:
        resolved = Path(path)
        if not resolved.is_absolute():
            return path
        return str(resolved.relative_to(cwd))
    except Exception:
        return path


def shorten_one_line(text: str, max_chars: int = 140) -> str:
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def command_preview(cmd: str) -> str:
    raw_lines = [line.rstrip() for line in cmd.replace("\r\n", "\n").replace("\r", "\n").splitlines()]
    lines = [line for line in raw_lines if line.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return f"Ran `{shorten_one_line(lines[0])}`"

    shown = lines[:DEFAULT_TOOL_COMMAND_PREVIEW_LINES]
    message = f"Ran `{shorten_one_line(shown[0])}`"
    for line in shown[1:]:
        message += "\n    " + clamp_replay_lines(line, DEFAULT_REPLAY_MAX_LINE_CHARS)
    hidden = len(lines) - len(shown)
    if hidden > 0:
        message += f"\n    ... {hidden} command lines hidden"
    return message


def output_preview(result_text: str) -> str:
    lines = clean_tool_output_lines(result_text)
    if not lines:
        return ""
    shown = lines[:DEFAULT_TOOL_OUTPUT_PREVIEW_LINES]
    rendered_lines = []
    for index, line in enumerate(shown):
        prefix = "  └ " if index == 0 else "    "
        rendered_lines.append(prefix + clamp_replay_lines(line, 140))
    rendered = "\n".join(rendered_lines)
    hidden = len(lines) - len(shown)
    if hidden > 0:
        rendered += f"\n    ... {hidden} output lines hidden"
    return rendered


def clean_tool_output_lines(result_text: str) -> list[str]:
    cleaned: list[str] = []
    in_fence = False
    for raw_line in result_text.strip().splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and (
            stripped.startswith("Exit code:")
            or stripped.startswith("Chunk ID:")
            or stripped.startswith("Process exited with code")
            or stripped.startswith("Original token count:")
            or stripped.startswith("Command output:")
            or stripped == "Output:"
            or stripped.startswith("Wall time:")
            or stripped.startswith("Command completed in ")
            or stripped.startswith("Shell state ")
        ):
            continue
        cleaned.append(line)
    return cleaned


def extract_output_body(result_text: str) -> str:
    text = result_text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return ""
    for marker in ("\nOutput:\n", "\nCommand output:\n", "Output:\n", "Command output:\n"):
        if marker in text:
            text = text.split(marker, 1)[1].lstrip("\n").rstrip("\n")
            break
    else:
        text = text.strip("\n")
    lines = []
    in_fence = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and (
            stripped.startswith("Command completed in ")
            or stripped.startswith("Shell state ")
        ):
            continue
        lines.append(raw.rstrip())
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def explicit_exit_code(result_text: str) -> int | None:
    match = re.search(r"(?:Exit code:|Process exited with code)\s*(-?\d+)", result_text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def parse_exit_code(result_text: str, default: int = 0) -> int:
    code = explicit_exit_code(result_text)
    if code is not None:
        return code
    lowered = result_text.casefold()
    if "failed" in lowered or "error" in lowered:
        return 1
    return default


def parse_wall_time(result_text: str, default: str = "0.0 seconds") -> str:
    match = re.search(r"Wall time:\s*([^\n]+)", result_text)
    if match:
        return match.group(1).strip()
    match = re.search(r"Command completed in\s+(\d+)\s*ms", result_text)
    if match:
        millis = int(match.group(1))
        return f"{millis / 1000:.1f} seconds"
    return default


def codex_exec_output(result_text: str, call_id: str) -> str:
    if result_text.startswith("Chunk ID:") and "Process exited with code" in result_text:
        return result_text
    body = extract_output_body(result_text)
    code = parse_exit_code(result_text)
    wall_time = parse_wall_time(result_text)
    chunk_id = re.sub(r"[^0-9A-Fa-f]", "", call_id)[-6:] or "000000"
    return (
        f"Chunk ID: {chunk_id}\n"
        f"Wall time: {wall_time}\n"
        f"Process exited with code {code}\n"
        f"Original token count: {estimate_tokens(body)}\n"
        "Output:\n"
        f"{body}"
    )


def patch_result_success(result_text: str) -> bool:
    lowered = result_text.casefold()
    if any(marker in lowered for marker in PATCH_FAILURE_MARKERS):
        return False
    code = explicit_exit_code(result_text)
    if code is not None:
        return code == 0
    stripped = lowered.strip()
    return bool(stripped) and any(marker in stripped for marker in PATCH_SUCCESS_MARKERS)


def diff_stats_and_preview(change: dict[str, Any]) -> tuple[int, int, list[str], int]:
    added = 0
    removed = 0
    preview: list[str] = []
    total_preview_lines = 0
    if change.get("type") == "add" and isinstance(change.get("content"), str):
        lines = ["+" + line for line in change["content"].splitlines()]
        added = len(lines)
        total_preview_lines = len(lines)
        preview = lines[:DEFAULT_PATCH_PREVIEW_LINES]
        return added, removed, preview, max(0, total_preview_lines - len(preview))
    if change.get("type") == "delete" and isinstance(change.get("content"), str):
        lines = ["-" + line for line in change["content"].splitlines()]
        removed = len(lines)
        total_preview_lines = len(lines)
        preview = lines[:DEFAULT_PATCH_PREVIEW_LINES]
        return added, removed, preview, max(0, total_preview_lines - len(preview))
    diff = change.get("unified_diff")
    if not isinstance(diff, str):
        return added, removed, preview, 0
    diff_lines: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
            diff_lines.append(line)
        elif line.startswith("-"):
            removed += 1
            diff_lines.append(line)
        elif line.startswith(" "):
            diff_lines.append(line)
    total_preview_lines = len(diff_lines)
    preview = diff_lines[:DEFAULT_PATCH_PREVIEW_LINES]
    return added, removed, preview, max(0, total_preview_lines - len(preview))


def patch_replay_message(changes: dict[str, Any] | None, cwd: Path) -> str | None:
    if not changes:
        return None
    messages: list[str] = []
    for raw_path, change in changes.items():
        if not isinstance(change, dict):
            continue
        path = display_path(raw_path, cwd)
        added, removed, preview, hidden = diff_stats_and_preview(change)
        if added == 0 and removed == 0 and not preview:
            continue
        summary = f"Edited `{path}`"
        if added or removed:
            summary += f" (+{added} -{removed})"
        if preview:
            snippet = "\n".join(preview)
            summary += f"\n\n```diff\n{snippet}\n```"
            if hidden:
                summary += f"\n... truncated ({hidden} more lines)"
        messages.append(summary)
    return "\n\n".join(messages) if messages else None


def patch_failure_replay_message(result_text: str) -> str | None:
    preview = output_preview(result_text)
    if not preview:
        return "Patch failed"
    return "Patch failed\n" + preview


def web_search_replay_message(args: Any, result_text: str = "") -> str | None:
    action = web_search_action(args)
    if not action:
        return None
    if action.get("type") == "open_page":
        url = action.get("url")
        message = f"Opened web page `{shorten_one_line(str(url or ''))}`"
    else:
        query = action.get("query")
        message = f"Searched web for `{shorten_one_line(str(query or ''))}`"
    preview = output_preview(result_text)
    if preview:
        message += "\n" + preview
    return message


def web_search_result_context_text(result_text: str) -> str:
    body = extract_output_body(result_text)
    if not body:
        body = result_text.strip()
    if not body:
        return ""
    return "[Cursor WebSearch result]\n" + body


def wait_replay_message(args: Any, result_text: str) -> str | None:
    wait_args = wait_arguments(args)
    if not wait_args:
        return None
    ids = wait_args.get("ids") or []
    target = ", ".join(str(item) for item in ids) if ids else "task"
    message = f"Waited for `{shorten_one_line(target)}`"
    preview = output_preview(result_text)
    if preview:
        message += "\n" + preview
    return message


def diagnostic_replay_message(args: Any, result_text: str) -> str | None:
    lint_args = read_lints_arguments(args)
    if not lint_args:
        return None
    path = lint_args.get("file") or ", ".join(lint_args.get("paths", []))
    message = f"Read diagnostics for `{shorten_one_line(str(path))}`"
    preview = output_preview(result_text)
    if preview:
        message += "\n" + preview
    return message


def semantic_search_replay_message(args: Any, result_text: str) -> str | None:
    search_args = semantic_search_arguments(args)
    if not search_args:
        return None
    message = f"Semantic searched `{shorten_one_line(str(search_args['query']))}`"
    preview = output_preview(result_text)
    if preview:
        message += "\n" + preview
    return message


def command_replay_message(tool_name: str | None, args: Any, result_text: str, cwd: Path) -> str | None:
    cmd = shell_cmd_for_tool(tool_name, args, cwd).strip()
    if not cmd:
        return None
    message = command_preview(cmd)
    preview = output_preview(result_text)
    if preview:
        message += "\n" + preview
    return message


def patch_tool_stdout(changes: dict[str, Any] | None, cwd: Path) -> str:
    if not changes:
        return "Success. Updated the following files:\n"
    lines = ["Success. Updated the following files:"]
    for raw_path, change in changes.items():
        if not isinstance(change, dict):
            continue
        change_type = change.get("type")
        prefix = {"add": "A", "update": "M", "delete": "D"}.get(str(change_type), "M")
        lines.append(f"{prefix} {display_path(raw_path, cwd)}")
    return "\n".join(lines) + "\n"


def is_exec_tool(tool_name: str | None) -> bool:
    return (tool_name or "").casefold() in EXEC_TOOL_NAMES


def is_patch_tool(tool_name: str | None) -> bool:
    return (tool_name or "").casefold() in PATCH_TOOL_NAMES


def build_tool_call_events(
    entry: ExportEntry,
    *,
    call_id: str,
    turn_id: str | None,
    timestamp_ms: int,
    cwd: Path,
    emit_tool_events: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    tool_name = (entry.tool_name or "").casefold()
    patch, _ = patch_for_tool(entry.tool_name, entry.args, cwd)
    if patch:
        payload: dict[str, Any] = {
            "type": "custom_tool_call",
            "id": item_id("ctc"),
            "status": "completed",
            "call_id": call_id,
            "name": "apply_patch",
            "input": patch,
        }
        payload.update(turn_metadata(turn_id))
        return ([{"timestamp": utc_iso(timestamp_ms), "type": "response_item", "payload": payload}], "custom")

    action = web_search_action(entry.args) if tool_name == "websearch" else None
    if action is not None:
        query = action.get("query") or action.get("url") or ""
        event_payload = {
            "type": "web_search_end",
            "call_id": call_id,
            "query": query,
            "action": action,
        }
        payload = {
            "type": "web_search_call",
            "status": "completed",
            "action": action,
        }
        payload.update(turn_metadata(turn_id))
        events = [{"timestamp": utc_iso(timestamp_ms), "type": "response_item", "payload": payload}]
        if emit_tool_events:
            events.insert(0, {"timestamp": utc_iso(timestamp_ms), "type": "event_msg", "payload": event_payload})
        return (events, "web_search")

    if tool_name == "semanticsearch":
        args = semantic_search_arguments(entry.args)
        if args is not None:
            payload = {
                "type": "function_call",
                "id": item_id("fc"),
                "name": "semantic_search",
                "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
                "call_id": call_id,
            }
            payload.update(turn_metadata(turn_id))
            return ([{"timestamp": utc_iso(timestamp_ms), "type": "response_item", "payload": payload}], "function")

    if tool_name == "readlints":
        args = read_lints_arguments(entry.args)
        if args is not None:
            payload = {
                "type": "function_call",
                "id": item_id("fc"),
                "name": "mcp__omx_code_intel__lsp_diagnostics",
                "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
                "call_id": call_id,
            }
            payload.update(turn_metadata(turn_id))
            return ([{"timestamp": utc_iso(timestamp_ms), "type": "response_item", "payload": payload}], "function")

    if tool_name == "await":
        args = wait_arguments(entry.args)
        if args is not None:
            payload = {
                "type": "function_call",
                "id": item_id("fc"),
                "name": "wait",
                "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
                "call_id": call_id,
            }
            payload.update(turn_metadata(turn_id))
            return ([{"timestamp": utc_iso(timestamp_ms), "type": "response_item", "payload": payload}], "function")

    plan_args = update_plan_arguments(entry.args) if tool_name == "todowrite" else None
    if plan_args is not None:
        payload = {
            "type": "function_call",
            "id": item_id("fc"),
            "name": "update_plan",
            "arguments": json.dumps(plan_args, ensure_ascii=False, separators=(",", ":")),
            "call_id": call_id,
        }
        payload.update(turn_metadata(turn_id))
        return ([{"timestamp": utc_iso(timestamp_ms), "type": "response_item", "payload": payload}], "plan")

    args = exec_command_arguments(entry.tool_name, entry.args, cwd)
    name = "exec_command"
    call_kind = "function"
    if args is None:
        name = normalize_function_name(entry.tool_name)
        args = generic_tool_arguments(entry.args)
        call_kind = "generic"
    payload: dict[str, Any] = {
        "type": "function_call",
        "id": item_id("fc"),
        "name": name,
        "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
        "call_id": call_id,
    }
    payload.update(turn_metadata(turn_id))
    return ([{"timestamp": utc_iso(timestamp_ms), "type": "response_item", "payload": payload}], call_kind)


def build_tool_result_events(
    entry: ExportEntry,
    *,
    call_id: str,
    call_kind: str,
    turn_id: str | None,
    timestamp_ms: int,
    pending_args: Any,
    pending_tool_name: str | None,
    cwd: Path,
    emit_tool_events: bool = False,
) -> list[dict[str, Any]]:
    result_text = str(entry.result or "")
    if call_kind == "custom":
        _, changes = patch_for_tool(pending_tool_name, pending_args, cwd)
        success = patch_result_success(result_text)
        if not success:
            return [
                {
                    "timestamp": utc_iso(timestamp_ms),
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call_output",
                        "call_id": call_id,
                        "output": result_text,
                        **turn_metadata(turn_id),
                    },
                }
            ]
        stdout = patch_tool_stdout(changes, cwd) if success else ""
        events = [
            {
                "timestamp": utc_iso(timestamp_ms),
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": call_id,
                    "output": "Exit code: 0\nWall time: 0.4 seconds\nOutput:\n" + stdout,
                    **turn_metadata(turn_id),
                },
            },
        ]
        if emit_tool_events:
            events.insert(
                0,
                {
                    "timestamp": utc_iso(timestamp_ms),
                    "type": "event_msg",
                    "payload": {
                        "type": "patch_apply_end",
                        "call_id": call_id,
                        "turn_id": turn_id,
                        "stdout": stdout,
                        "stderr": "",
                        "success": True,
                        "changes": changes or {},
                        "status": "completed",
                    },
                },
            )
        return events
    if call_kind == "web_search":
        context_text = web_search_result_context_text(result_text)
        if not context_text:
            return []
        return [
            {
                "timestamp": utc_iso(timestamp_ms),
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": item_id("msg"),
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": context_text}],
                    "phase": "commentary",
                    **turn_metadata(turn_id),
                },
            }
        ]
    if call_kind == "plan":
        result_text = "Plan updated"
    elif call_kind == "function":
        result_text = codex_exec_output(result_text, call_id)
    return [
        {
            "timestamp": utc_iso(timestamp_ms),
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result_text,
                **turn_metadata(turn_id),
            },
        }
    ]


def build_events(
    session_id: str,
    entries: list[ExportEntry],
    title: str,
    cwd: Path,
    created_ms: int,
    defaults: ThreadDefaults,
    replay_mode: str = "split",
    tool_replay_mode: str = DEFAULT_TOOL_REPLAY_MODE,
) -> list[dict[str, Any]]:
    if replay_mode not in REPLAY_MODES:
        raise ValueError(f"unknown replay mode: {replay_mode}")
    if tool_replay_mode not in TOOL_REPLAY_MODES:
        raise ValueError(f"unknown tool replay mode: {tool_replay_mode}")
    created_iso = utc_iso(created_ms)
    events: list[dict[str, Any]] = [
        {
            "timestamp": created_iso,
            "type": "session_meta",
            "payload": {
                "session_id": session_id,
                "id": session_id,
                "timestamp": created_iso,
                "cwd": str(cwd),
                "originator": "codex-tui",
                "cli_version": defaults.cli_version,
                "source": "cli",
                "thread_source": "user",
                "model_provider": "openai",
                "cursor_to_codex_resume_version": EXPORT_SCHEMA_VERSION,
                "cursor_to_codex_resume_replay": replay_mode,
                "cursor_to_codex_resume_tool_replay": tool_replay_mode,
            },
        }
    ]

    offset = 1
    turn_id: str | None = None
    turn_started_ms = created_ms
    last_agent_message = ""
    input_tokens = 0
    output_tokens = 0
    call_map: dict[str, tuple[str, str, str | None, Any]] = {}
    call_index = 0

    def next_ms() -> int:
        nonlocal offset
        value = created_ms + offset * 10
        offset += 1
        return value

    def finish_turn() -> None:
        nonlocal turn_id, last_agent_message, input_tokens, output_tokens
        if not turn_id:
            return
        ts = next_ms()
        events.append(token_count_event(ts, input_tokens, output_tokens))
        complete_ts = next_ms()
        events.append(
            {
                "timestamp": utc_iso(complete_ts),
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "turn_id": turn_id,
                    "last_agent_message": last_agent_message,
                    "completed_at": complete_ts // 1000,
                    "duration_ms": max(1000, complete_ts - turn_started_ms),
                    "time_to_first_token_ms": 300,
                },
            }
        )
        turn_id = None
        last_agent_message = ""
        input_tokens = 0
        output_tokens = 0

    for entry in entries:
        if entry.kind == "user":
            finish_turn()
            turn_id = uuid7_like()
            turn_started_ms = next_ms()
            events.append(
                {
                    "timestamp": utc_iso(turn_started_ms),
                    "type": "event_msg",
                    "payload": {
                        "type": "task_started",
                        "turn_id": turn_id,
                        "started_at": turn_started_ms // 1000,
                        "model_context_window": DEFAULT_MODEL_CONTEXT_WINDOW,
                        "collaboration_mode_kind": "default",
                    },
                }
            )
            text = entry.text or ""
            replay_chunks = replay_texts(text, replay_mode)
            input_tokens += estimate_tokens(text)
            events.append(
                {
                    "timestamp": utc_iso(next_ms()),
                    "type": "turn_context",
                    "payload": turn_context_payload(
                        turn_id=turn_id,
                        cwd=cwd,
                        created_ms=turn_started_ms,
                        defaults=defaults,
                    ),
                }
            )
            user_ts = next_ms()
            events.append(
                {
                    "timestamp": utc_iso(user_ts),
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                        **turn_metadata(turn_id),
                    },
                }
            )
            for replay_text in replay_chunks:
                events.append(
                    {
                        "timestamp": utc_iso(user_ts),
                        "type": "event_msg",
                        "payload": {
                            "type": "user_message",
                            "message": replay_text,
                            "images": [],
                            "local_images": [],
                            "text_elements": [],
                        },
                    }
                )
            continue

        if not turn_id:
            turn_id = uuid7_like()
            turn_started_ms = next_ms()
            events.append(
                {
                    "timestamp": utc_iso(turn_started_ms),
                    "type": "event_msg",
                    "payload": {
                        "type": "task_started",
                        "turn_id": turn_id,
                        "started_at": turn_started_ms // 1000,
                        "model_context_window": DEFAULT_MODEL_CONTEXT_WINDOW,
                        "collaboration_mode_kind": "default",
                    },
                }
            )

        if entry.kind == "reasoning":
            # Cursor reasoning/redacted-reasoning is not an OpenAI encrypted
            # reasoning blob. Replaying it as encrypted_content causes
            # invalid_encrypted_content on the next Codex API request.
            continue

        if entry.kind == "assistant":
            text = entry.text or ""
            replay_chunks = replay_texts(text, replay_mode)
            phase = entry.phase or "final_answer"
            output_tokens += estimate_tokens(text)
            ts = next_ms()
            for replay_text in replay_chunks:
                events.append(
                    {
                        "timestamp": utc_iso(ts),
                        "type": "event_msg",
                        "payload": {
                            "type": "agent_message",
                            "message": replay_text,
                            "phase": phase,
                            "memory_citation": None,
                        },
                    }
                )
            events.append(
                {
                    "timestamp": utc_iso(ts),
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": item_id("msg"),
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                        "phase": phase,
                        **turn_metadata(turn_id),
                    },
                }
            )
            last_agent_message = replay_chunks[-1] if replay_mode == "compact" else ""
            continue

        if entry.kind == "tool_call":
            call_index += 1
            call_id = stable_call_id(entry.tool_call_id, entry.tool_name, call_index)
            tool_events, call_kind = build_tool_call_events(
                entry,
                call_id=call_id,
                turn_id=turn_id,
                timestamp_ms=next_ms(),
                cwd=cwd,
                emit_tool_events=tool_replay_mode == "compact",
            )
            events.extend(tool_events)
            if entry.tool_call_id:
                call_map[entry.tool_call_id] = (call_id, call_kind, entry.tool_name, entry.args)
            continue

        if entry.kind == "tool_result":
            mapped = call_map.get(entry.tool_call_id or "")
            if not mapped:
                call_index += 1
                mapped = (
                    stable_call_id(entry.tool_call_id, entry.tool_name, call_index),
                    "function",
                    entry.tool_name,
                    None,
                )
            call_id, call_kind, pending_tool_name, pending_args = mapped
            if call_kind == "custom":
                _, changes = patch_for_tool(pending_tool_name, pending_args, cwd)
                result_text = str(entry.result or "")
                replay_message = (
                    patch_replay_message(changes, cwd)
                    if patch_result_success(result_text)
                    else patch_failure_replay_message(result_text)
                )
            elif call_kind == "web_search":
                replay_message = web_search_replay_message(
                    pending_args if pending_args is not None else entry.args,
                    str(entry.result or ""),
                )
            else:
                replay_args = pending_args if pending_args is not None else entry.args
                pending_name = (pending_tool_name or entry.tool_name or "").casefold()
                result_text = str(entry.result or "")
                if pending_name == "await":
                    replay_message = wait_replay_message(replay_args, result_text)
                elif pending_name == "readlints":
                    replay_message = diagnostic_replay_message(replay_args, result_text)
                elif pending_name == "semanticsearch":
                    replay_message = semantic_search_replay_message(replay_args, result_text)
                else:
                    replay_message = command_replay_message(
                        pending_tool_name or entry.tool_name,
                        replay_args,
                        result_text,
                        cwd,
                    )
            if tool_replay_mode == "compact" and replay_message:
                events.append(
                    {
                        "timestamp": utc_iso(next_ms()),
                        "type": "event_msg",
                        "payload": {
                            "type": "agent_message",
                            "message": replay_message,
                            "phase": "commentary",
                            "memory_citation": None,
                        },
                    }
                )
            events.extend(
                build_tool_result_events(
                    entry,
                    call_id=call_id,
                    call_kind=call_kind,
                    turn_id=turn_id,
                    timestamp_ms=next_ms(),
                    pending_args=pending_args,
                    pending_tool_name=pending_tool_name or entry.tool_name,
                    cwd=cwd,
                    emit_tool_events=tool_replay_mode == "compact",
                )
            )
            input_tokens += estimate_tokens(str(entry.result or ""))

    finish_turn()
    return events


def find_state_db(codex_home: Path) -> Path | None:
    dbs = []
    for path in codex_home.glob("state_*.sqlite"):
        try:
            suffix = int(path.stem.split("_", 1)[1])
        except Exception:
            suffix = -1
        dbs.append((suffix, path.stat().st_mtime, path))
    if not dbs:
        return None
    return sorted(dbs)[-1][2]


def thread_exists(state_db: Path | None, session_id: str) -> bool:
    if not state_db or not state_db.exists():
        return False
    try:
        con = sqlite3.connect(state_db)
        row = con.execute("select 1 from threads where id = ?", (session_id,)).fetchone()
        con.close()
        return row is not None
    except sqlite3.Error:
        return False


def thread_rollout_path(state_db: Path | None, session_id: str) -> str | None:
    if not state_db or not state_db.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
        row = con.execute("select rollout_path from threads where id = ?", (session_id,)).fetchone()
        con.close()
    except sqlite3.Error:
        return None
    if row and isinstance(row[0], str) and row[0]:
        return row[0]
    return None


def detect_codex_cli_version() -> str:
    try:
        proc = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return "0.142.3"
    text = (proc.stdout or proc.stderr or "").strip()
    match = re.search(r"(\d+\.\d+\.\d+)", text)
    return match.group(1) if match else "0.142.3"


def read_thread_defaults(state_db: Path | None, exclude_session_id: str) -> ThreadDefaults:
    fallback = ThreadDefaults(cli_version=detect_codex_cli_version())
    if not state_db or not state_db.exists():
        return fallback
    try:
        con = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
        row = con.execute(
            """
            select cli_version, model, reasoning_effort, approval_mode
            from threads
            where id != ?
              and coalesce(cli_version, '') != 'cursor-to-codex-resume'
            order by coalesce(updated_at_ms, updated_at * 1000) desc
            limit 1
            """,
            (exclude_session_id,),
        ).fetchone()
        con.close()
    except sqlite3.Error:
        return fallback
    if not row:
        return fallback
    return ThreadDefaults(
        cli_version=row[0] or fallback.cli_version,
        model=row[1],
        reasoning_effort=row[2],
        approval_mode=row[3] or fallback.approval_mode,
    )


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def upsert_thread(
    state_db: Path | None,
    session_id: str,
    rollout_path: Path,
    title: str,
    cwd: Path,
    created_ms: int,
    updated_ms: int,
    first_user: str,
    preview: str,
    defaults: ThreadDefaults,
    tokens_used: int,
) -> bool:
    if not state_db or not state_db.exists():
        return False
    sandbox_policy = json.dumps({"type": "disabled"}, separators=(",", ":"))
    values: dict[str, Any] = {
        "id": session_id,
        "rollout_path": str(rollout_path),
        "created_at": created_ms // 1000,
        "updated_at": updated_ms // 1000,
        "source": "cli",
        "model_provider": "openai",
        "cwd": str(cwd),
        "title": title,
        "sandbox_policy": sandbox_policy,
        "approval_mode": defaults.approval_mode,
        "tokens_used": tokens_used,
        "has_user_event": 1,
        "archived": 0,
        "cli_version": defaults.cli_version,
        "first_user_message": first_user,
        "memory_mode": "enabled",
        "model": defaults.model,
        "reasoning_effort": defaults.reasoning_effort,
        "thread_source": "user",
        "preview": preview,
        "recency_at": updated_ms // 1000,
        "recency_at_ms": updated_ms,
        "created_at_ms": created_ms,
        "updated_at_ms": updated_ms,
    }
    try:
        con = sqlite3.connect(state_db, timeout=10)
        columns = [row[1] for row in con.execute("pragma table_info(threads)")]
        selected = [column for column in columns if column in values]
        placeholders = ",".join("?" for _ in selected)
        quoted = ",".join(f'"{column}"' for column in selected)
        sql = f'insert or replace into threads ({quoted}) values ({placeholders})'
        con.execute(sql, [values[column] for column in selected])
        con.commit()
        con.close()
        return True
    except sqlite3.Error as exc:
        eprint(f"warning: failed to update {state_db}: {exc}")
        return False


def update_session_index(codex_home: Path, session_id: str, title: str, updated_ms: int) -> bool:
    path = codex_home / "session_index.jsonl"
    rows: list[dict[str, Any]] = []
    changed = False
    if path.exists():
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("id") == session_id:
                        row["thread_name"] = title
                        row["updated_at"] = utc_iso(updated_ms)
                        changed = True
                    rows.append(row)
        except OSError:
            return False
    if changed:
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"id": session_id, "thread_name": title, "updated_at": utc_iso(updated_ms)},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
    return True


def import_candidate(args: argparse.Namespace) -> dict[str, Any]:
    cursor_home = expand(args.cursor_home)
    codex_home = expand(args.codex_home)
    candidates = discover_candidates(cursor_home)
    current_cwd = expand(args.cwd) if args.cwd else maybe_resolve(Path.cwd())
    candidate, selected_by = select_candidate(
        candidates,
        args.chat,
        current_cwd,
    )
    entries = extract_entries(candidate, args.include_tools)
    visible_messages = sum(1 for entry in entries if entry.kind in {"user", "assistant"} and entry.text)
    if not entries or visible_messages == 0:
        raise SystemExit(
            "INFO: The selected Cursor session has no conversation to export after removing the handoff command. "
            "No user/assistant messages remained in the selected session. "
            f"selected_by={selected_by!r}, cwd={str(current_cwd)!r}, "
            f"chat_id={candidate.chat_id!r}, title={candidate.title!r}, "
            f"updated_at={candidate_updated_at(candidate)!r}. "
            "No Codex session was written. "
            "Open this skill from a Cursor session that already has real conversation history, "
            "or use --list and --chat <id> for a manual import."
        )
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    created_ms = candidate.created_ms or candidate.updated_ms or now_ms
    updated_ms = candidate.updated_ms or created_ms
    cwd = current_cwd if args.cwd else candidate_cwd(candidate, current_cwd)
    first_user = next((entry.text for entry in entries if entry.kind == "user" and entry.text), "")
    title = args.title or candidate.title or first_user[:120] or candidate.chat_id
    first_user = first_user or title
    preview = first_user[:4000]
    session_id = session_id_for(candidate, args.new)
    year, month, day, filename = local_rollout_parts(created_ms, session_id)
    rollout_path = codex_home / "sessions" / year / month / day / filename
    state_db = find_state_db(codex_home)
    defaults = read_thread_defaults(state_db, session_id)
    replay_mode = "full" if args.full_replay else "compact" if args.compact_replay else "split"
    events = build_events(
        session_id,
        entries,
        title,
        cwd,
        created_ms,
        defaults,
        replay_mode,
        args.tool_replay,
    )
    tokens_used = sum(estimate_tokens(entry.text) for entry in entries if entry.text)

    summary = {
        "cursor_chat_id": candidate.chat_id,
        "title": title,
        "session_id": session_id,
        "entries": len(entries),
        "visible_messages": visible_messages,
        "tool_calls": sum(1 for entry in entries if entry.kind == "tool_call"),
        "tool_results": sum(1 for entry in entries if entry.kind == "tool_result"),
        "cwd": str(cwd),
        "rollout_path": str(rollout_path),
        "state_db": str(state_db) if state_db else None,
        "transcript_path": str(candidate.transcript_path) if candidate.transcript_path else None,
        "store_db": str(candidate.store_db) if candidate.store_db else None,
        "selected_by": selected_by,
        "selected_candidate": candidate_debug_summary(candidate),
        "replay": replay_mode,
        "tool_replay": args.tool_replay,
        "codex_defaults": {
            "cli_version": defaults.cli_version,
            "model": defaults.model,
            "reasoning_effort": defaults.reasoning_effort,
            "approval_mode": defaults.approval_mode,
        },
    }
    if args.dry_run:
        return summary

    needs_repair = rollout_needs_rewrite(rollout_path, replay_mode, args.tool_replay)
    existing_thread_rollout = thread_rollout_path(state_db, session_id)
    already_current = rollout_path.exists() and (
        not state_db
        or not state_db.exists()
        or existing_thread_rollout == str(rollout_path)
    )
    if (
        not args.force
        and not args.new
        and not needs_repair
        and already_current
    ):
        summary["status"] = "already-imported"
        return summary

    codex_home.mkdir(parents=True, exist_ok=True)
    write_jsonl(rollout_path, events)
    summary["state_updated"] = upsert_thread(
        state_db,
        session_id,
        rollout_path,
        title,
        cwd,
        created_ms,
        updated_ms,
        first_user[:16000],
        preview,
        defaults,
        tokens_used,
    )
    summary["session_index_updated"] = update_session_index(codex_home, session_id, title, updated_ms)
    if state_db and state_db.exists() and not summary["state_updated"]:
        summary["status"] = "registration-failed"
        summary["error"] = (
            f"Failed to update Codex state database {state_db}. "
            "The rollout file was written, but Codex resume may not list this session. "
            "Fix the state database compatibility issue and rerun the importer."
        )
        return summary
    if (not state_db or not state_db.exists()) and not summary["session_index_updated"]:
        summary["status"] = "registration-failed"
        summary["error"] = (
            "Failed to update Codex session_index.jsonl and no Codex state database was available. "
            "The rollout file was written, but Codex resume may not list this session."
        )
        return summary
    summary["status"] = "repaired" if needs_repair else "imported"
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import Cursor Agent chats into Codex CLI resume state."
    )
    parser.add_argument("--cursor-home", default="~/.cursor", help="Cursor home directory")
    parser.add_argument(
        "--codex-home",
        default=os.environ.get("CODEX_HOME", "~/.codex"),
        help="Codex home directory",
    )
    parser.add_argument("--chat", default="current", help="current, latest, chat id prefix, title, path")
    parser.add_argument("--title", help="Override imported Codex thread title")
    parser.add_argument("--cwd", help="Override current workspace and imported Codex thread cwd")
    parser.add_argument("--list", action="store_true", help="List importable Cursor chats")
    parser.add_argument("--dry-run", action="store_true", help="Print planned import only")
    parser.add_argument("--force", action="store_true", help="Replace an existing deterministic import")
    parser.add_argument("--new", action="store_true", help="Create a new Codex session id")
    parser.add_argument(
        "--include-tools",
        action="store_true",
        default=True,
        help="Include Cursor tool calls/results in the Codex rollout (default)",
    )
    parser.add_argument(
        "--no-tools",
        action="store_false",
        dest="include_tools",
        help="Import only visible user/assistant text",
    )
    parser.add_argument(
        "--compact-replay",
        action="store_true",
        help="Shorten long Codex TUI replay messages while keeping full model context",
    )
    parser.add_argument(
        "--full-replay",
        action="store_true",
        help="Keep raw one-message TUI replay blocks for debugging Codex resume rendering",
    )
    parser.add_argument(
        "--tool-replay",
        choices=sorted(TOOL_REPLAY_MODES),
        default=DEFAULT_TOOL_REPLAY_MODE,
        help=(
            "Tool display replay mode: none for Codex-resume-like display (default) "
            "or compact text summaries. "
            "Stock Codex CLI 0.142.3 does not expose a stable no-rerun API for "
            "injecting native Ran/Edited TUI widgets into closed-session resume replay."
        ),
    )
    args = parser.parse_args(argv)

    if args.list:
        candidates = [c for c in discover_candidates(expand(args.cursor_home)) if c.transcript_path or c.store_db]
        if not candidates:
            print("No Cursor chats with transcript or store.db were found.")
            return 0
        for index, candidate in enumerate(candidates, 1):
            print(format_candidate(index, candidate))
        return 0

    summary = import_candidate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("status") in {"imported", "repaired"}:
        print("\nRun: codex resume --all")
    return 1 if summary.get("status") == "registration-failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
