#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPORTER = ROOT / "scripts" / "import_cursor_chat.py"


def parse_summary(stdout: str) -> dict:
    summary, _ = json.JSONDecoder().raw_decode(stdout)
    assert isinstance(summary, dict)
    return summary


def create_codex_state_db(codex_home: Path) -> Path:
    codex_home.mkdir(parents=True, exist_ok=True)
    state_db = codex_home / "state_5.sqlite"
    con = sqlite3.connect(state_db)
    con.execute(
        """
        CREATE TABLE threads (
            id TEXT PRIMARY KEY,
            rollout_path TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            source TEXT NOT NULL,
            model_provider TEXT NOT NULL,
            cwd TEXT NOT NULL,
            title TEXT NOT NULL,
            sandbox_policy TEXT NOT NULL,
            approval_mode TEXT NOT NULL,
            tokens_used INTEGER NOT NULL DEFAULT 0,
            has_user_event INTEGER NOT NULL DEFAULT 0,
            archived INTEGER NOT NULL DEFAULT 0,
            archived_at INTEGER,
            git_sha TEXT,
            git_branch TEXT,
            git_origin_url TEXT,
            cli_version TEXT NOT NULL DEFAULT '',
            first_user_message TEXT NOT NULL DEFAULT '',
            agent_nickname TEXT,
            agent_role TEXT,
            memory_mode TEXT NOT NULL DEFAULT 'enabled',
            model TEXT,
            reasoning_effort TEXT,
            agent_path TEXT,
            created_at_ms INTEGER,
            updated_at_ms INTEGER,
            thread_source TEXT,
            preview TEXT NOT NULL DEFAULT '',
            recency_at INTEGER NOT NULL DEFAULT 0,
            recency_at_ms INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    con.commit()
    con.close()
    return state_db


class EndToEndImportTests(unittest.TestCase):
    def test_transcript_import_writes_resume_surfaces_without_user_home_writes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="c2c-e2e-") as tmp:
            root = Path(tmp)
            cursor_home = root / "cursor"
            codex_home = root / "codex"
            cwd = root / "repo"
            chat_id = "11111111-2222-3333-4444-555555555555"
            transcript = (
                cursor_home
                / "projects"
                / "tmp-c2c-e2e-repo"
                / "agent-transcripts"
                / chat_id
                / f"{chat_id}.jsonl"
            )
            cwd.mkdir(parents=True)
            transcript.parent.mkdir(parents=True)
            state_db = create_codex_state_db(codex_home)
            transcript.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "role": "user",
                                "message": {
                                    "content": "Please summarize this table.\n\n| 파일 | 역할 |\n|---|---|\n| SKILL.md | Codex 스킬 규칙 |"
                                },
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "role": "assistant",
                                "message": {
                                    "content": "정리했습니다.\n\n```mermaid\nflowchart TD\nA-->B\n```"
                                },
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(IMPORTER),
                    "--cursor-home",
                    str(cursor_home),
                    "--codex-home",
                    str(codex_home),
                    "--chat",
                    chat_id,
                    "--cwd",
                    str(cwd),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            summary = parse_summary(proc.stdout)
            rollout_path = Path(summary["rollout_path"])
            self.assertTrue(rollout_path.exists())
            self.assertTrue(str(rollout_path).startswith(str(codex_home)))
            self.assertEqual(summary["visible_messages"], 2)
            self.assertEqual(summary["cwd"], str(cwd))
            self.assertTrue((codex_home / "session_index.jsonl").exists())
            self.assertTrue(summary["state_updated"])
            self.assertEqual(summary["state_db"], str(state_db))

            events = [
                json.loads(line)
                for line in rollout_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            payloads = [event.get("payload", {}) for event in events]
            event_messages = [
                payload.get("message")
                for event in events
                for payload in [event.get("payload", {})]
                if event.get("type") == "event_msg"
                and payload.get("type") in {"user_message", "agent_message"}
            ]
            response_messages = [
                payload
                for event in events
                for payload in [event.get("payload", {})]
                if event.get("type") == "response_item"
                and payload.get("type") == "message"
            ]

            self.assertEqual(payloads[0]["session_id"], summary["session_id"])
            self.assertTrue(any(payload.get("type") == "task_started" for payload in payloads))
            self.assertTrue(any(payload.get("type") == "task_complete" for payload in payloads))
            self.assertTrue(any("┌" in message for message in event_messages if message))
            self.assertTrue(any("```mermaid" in message for message in event_messages if message))
            self.assertTrue(
                any(
                    "| 파일 | 역할 |" in content.get("text", "")
                    for payload in response_messages
                    for content in payload.get("content", [])
                )
            )
            con = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
            row = con.execute(
                """
                select id, rollout_path, source, cwd, title, has_user_event,
                       archived, thread_source, preview
                from threads where id = ?
                """,
                (summary["session_id"],),
            ).fetchone()
            con.close()
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row[0], summary["session_id"])
            self.assertEqual(row[1], str(rollout_path))
            self.assertEqual(row[2], "cli")
            self.assertEqual(row[3], str(cwd))
            self.assertEqual(row[4], summary["title"])
            self.assertEqual(row[5], 1)
            self.assertEqual(row[6], 0)
            self.assertEqual(row[7], "user")
            self.assertIn("Please summarize this table.", row[8])

    def test_empty_selected_transcript_refuses_without_writing_session(self) -> None:
        with tempfile.TemporaryDirectory(prefix="c2c-empty-e2e-") as tmp:
            root = Path(tmp)
            cursor_home = root / "cursor"
            codex_home = root / "codex"
            cwd = root / "repo"
            chat_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            transcript = (
                cursor_home
                / "projects"
                / "tmp-c2c-empty-repo"
                / "agent-transcripts"
                / chat_id
                / f"{chat_id}.jsonl"
            )
            cwd.mkdir(parents=True)
            transcript.parent.mkdir(parents=True)
            transcript.write_text(
                json.dumps(
                    {
                        "role": "user",
                        "message": {"content": "$cursor-to-codex-resume"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(IMPORTER),
                    "--cursor-home",
                    str(cursor_home),
                    "--codex-home",
                    str(codex_home),
                    "--chat",
                    chat_id,
                    "--cwd",
                    str(cwd),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("INFO: The selected Cursor session has no conversation", proc.stderr)
            self.assertFalse((codex_home / "sessions").exists())
            self.assertFalse((codex_home / "session_index.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
