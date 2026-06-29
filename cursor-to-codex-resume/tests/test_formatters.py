#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPORTER_PATH = ROOT / "scripts" / "import_cursor_chat.py"


def load_importer():
    spec = importlib.util.spec_from_file_location("cursor_to_codex_importer", IMPORTER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FormatterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.importer = load_importer()

    def test_markdown_table_visible_replay_uses_terminal_table(self) -> None:
        text = "| 파일 | 역할 |\n|---|---|\n| SKILL.md | Codex 스킬 규칙 |\n"
        rendered = self.importer.render_markdown_tables_for_replay(text)

        self.assertIn("┌", rendered)
        self.assertIn("│ 파일", rendered)
        self.assertIn("Codex 스킬 규칙", rendered)
        self.assertNotIn("|---|", rendered)

    def test_markdown_table_inside_fence_is_preserved(self) -> None:
        text = "```markdown\n| a | b |\n|---|---|\n```\n"
        rendered = self.importer.render_markdown_tables_for_replay(text)

        self.assertEqual(rendered, text)

    def test_multiline_command_replay_keeps_leading_command_lines(self) -> None:
        message = self.importer.command_replay_message(
            "shell",
            {
                "command": "\n".join(
                    [
                        "set -e",
                        "work=/tmp/c2c-empty-v13-ws",
                        "cursor_home=$(mktemp -d)",
                        'mkdir -p "$cursor_home"',
                        "python import_cursor_chat.py --dry-run",
                    ]
                )
            },
            "Exit code: 1\nWall time: 0.1 seconds\nOutput:\nstatus=1\nINFO: empty\n",
            Path("/home/yjh/bypass"),
        )

        self.assertIsNotNone(message)
        assert message is not None
        self.assertIn("Ran `set -e`", message)
        self.assertIn("    work=/tmp/c2c-empty-v13-ws", message)
        self.assertIn("    cursor_home=$(mktemp -d)", message)
        self.assertIn("  └ status=1", message)
        self.assertIn("    INFO: empty", message)
        self.assertNotIn("Exit code:", message)
        self.assertNotIn("Wall time:", message)

    def test_patch_replay_message_uses_relative_path_and_diff_stats(self) -> None:
        message = self.importer.patch_replay_message(
            {
                "/home/yjh/bypass/lib.py": {
                    "type": "update",
                    "unified_diff": "\n".join(
                        [
                            "--- a/lib.py",
                            "+++ b/lib.py",
                            "@@ -1,3 +1,3 @@",
                            "-old = True",
                            "+new = True",
                            " unchanged",
                        ]
                    ),
                }
            },
            Path("/home/yjh/bypass"),
        )

        self.assertIsNotNone(message)
        assert message is not None
        self.assertIn("Edited `lib.py` (+1 -1)", message)
        self.assertIn("```diff", message)
        self.assertIn("-old = True", message)
        self.assertIn("+new = True", message)

    def test_build_events_keeps_model_context_and_compact_tool_replay(self) -> None:
        entries = [
            self.importer.ExportEntry(kind="user", text="do it"),
            self.importer.ExportEntry(
                kind="tool_call",
                tool_name="shell",
                tool_call_id="shell-1",
                args={"command": "set -e\necho hi"},
            ),
            self.importer.ExportEntry(
                kind="tool_result",
                tool_call_id="shell-1",
                result="Output:\nhi\n",
            ),
            self.importer.ExportEntry(
                kind="tool_call",
                tool_name="write",
                tool_call_id="write-1",
                args={"path": "/home/yjh/bypass/hello.txt", "contents": "hello\n"},
            ),
            self.importer.ExportEntry(
                kind="tool_result",
                tool_call_id="write-1",
                result="ok",
            ),
            self.importer.ExportEntry(kind="assistant", text="done"),
        ]

        events = self.importer.build_events(
            "11111111-1111-1111-1111-111111111111",
            entries,
            "test",
            Path("/home/yjh/bypass"),
            1_700_000_000_000,
            self.importer.ThreadDefaults(cli_version="0.142.3", model="gpt-5.5"),
        )
        agent_messages = [
            event["payload"]["message"]
            for event in events
            if event.get("type") == "event_msg"
            and event.get("payload", {}).get("type") == "agent_message"
        ]
        response_types = [
            event["payload"]["type"]
            for event in events
            if event.get("type") == "response_item"
        ]

        self.assertTrue(any(message.startswith("Ran `set -e`") for message in agent_messages))
        self.assertTrue(any("Edited `hello.txt` (+1 -0)" in message for message in agent_messages))
        self.assertIn("function_call", response_types)
        self.assertIn("function_call_output", response_types)
        self.assertIn("custom_tool_call", response_types)
        self.assertIn("custom_tool_call_output", response_types)

        function_calls = [
            event["payload"]
            for event in events
            if event.get("type") == "response_item"
            and event.get("payload", {}).get("type") == "function_call"
        ]
        shell_call = next(payload for payload in function_calls if payload["name"] == "exec_command")
        shell_args = self.importer.json.loads(shell_call["arguments"])
        self.assertEqual(shell_args["cmd"], "set -e\necho hi")
        self.assertEqual(shell_args["workdir"], "/home/yjh/bypass")
        self.assertEqual(shell_args["max_output_tokens"], self.importer.DEFAULT_EXEC_MAX_OUTPUT_TOKENS)

        function_outputs = [
            event["payload"]["output"]
            for event in events
            if event.get("type") == "response_item"
            and event.get("payload", {}).get("type") == "function_call_output"
        ]
        self.assertTrue(any(output.startswith("Chunk ID: ") for output in function_outputs))
        self.assertTrue(any("Process exited with code 0" in output for output in function_outputs))

    def test_cursor_todowrite_maps_to_codex_update_plan(self) -> None:
        events = self.importer.build_events(
            "11111111-1111-1111-1111-111111111111",
            [
                self.importer.ExportEntry(kind="user", text="track it"),
                self.importer.ExportEntry(
                    kind="tool_call",
                    tool_name="TodoWrite",
                    tool_call_id="todo-1",
                    args={
                        "todos": [
                            {"content": "Inspect native rollout", "status": "in_progress"},
                            {"content": "Patch importer", "status": "pending"},
                        ]
                    },
                ),
                self.importer.ExportEntry(
                    kind="tool_result",
                    tool_call_id="todo-1",
                    result="ok",
                ),
            ],
            "test",
            Path("/home/yjh/bypass"),
            1_700_000_000_000,
            self.importer.ThreadDefaults(cli_version="0.142.3"),
        )

        call = next(
            event["payload"]
            for event in events
            if event.get("type") == "response_item"
            and event.get("payload", {}).get("type") == "function_call"
        )
        output = next(
            event["payload"]["output"]
            for event in events
            if event.get("type") == "response_item"
            and event.get("payload", {}).get("type") == "function_call_output"
        )

        self.assertEqual(call["name"], "update_plan")
        args = self.importer.json.loads(call["arguments"])
        self.assertEqual(args["plan"][0]["step"], "Inspect native rollout")
        self.assertEqual(args["plan"][0]["status"], "in_progress")
        self.assertEqual(output, "Plan updated")

    def test_cursor_delete_maps_to_apply_patch_delete_file(self) -> None:
        events = self.importer.build_events(
            "11111111-1111-1111-1111-111111111111",
            [
                self.importer.ExportEntry(kind="user", text="delete it"),
                self.importer.ExportEntry(
                    kind="tool_call",
                    tool_name="Delete",
                    tool_call_id="delete-1",
                    args={"path": "/home/yjh/bypass/old.txt"},
                ),
                self.importer.ExportEntry(
                    kind="tool_result",
                    tool_call_id="delete-1",
                    result="Successfully deleted file: /home/yjh/bypass/old.txt",
                ),
            ],
            "test",
            Path("/home/yjh/bypass"),
            1_700_000_000_000,
            self.importer.ThreadDefaults(cli_version="0.142.3"),
        )

        call = next(
            event["payload"]
            for event in events
            if event.get("type") == "response_item"
            and event.get("payload", {}).get("type") == "custom_tool_call"
        )
        patch_end = next(
            event["payload"]
            for event in events
            if event.get("type") == "event_msg"
            and event.get("payload", {}).get("type") == "patch_apply_end"
        )

        self.assertEqual(call["name"], "apply_patch")
        self.assertIn("*** Delete File: /home/yjh/bypass/old.txt", call["input"])
        self.assertTrue(patch_end["success"])
        self.assertEqual(patch_end["changes"]["/home/yjh/bypass/old.txt"]["type"], "delete")

    def test_cursor_write_to_git_tracked_file_maps_to_update_patch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="c2c-write-git-") as tmp:
            repo = Path(tmp)
            target = repo / "app.py"
            target.write_text("old = True\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
            subprocess.run(["git", "add", "app.py"], cwd=repo, check=True, capture_output=True, text=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Cursor To Codex Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-m",
                    "seed app",
                ],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )

            events = self.importer.build_events(
                "11111111-1111-1111-1111-111111111111",
                [
                    self.importer.ExportEntry(kind="user", text="rewrite it"),
                    self.importer.ExportEntry(
                        kind="tool_call",
                        tool_name="Write",
                        tool_call_id="write-1",
                        args={"path": str(target), "contents": "new = True\n"},
                    ),
                    self.importer.ExportEntry(kind="tool_result", tool_call_id="write-1", result="ok"),
                ],
                "test",
                repo,
                1_700_000_000_000,
                self.importer.ThreadDefaults(cli_version="0.142.3"),
            )

            call = next(
                event["payload"]
                for event in events
                if event.get("type") == "response_item"
                and event.get("payload", {}).get("type") == "custom_tool_call"
            )
            patch_end = next(
                event["payload"]
                for event in events
                if event.get("type") == "event_msg"
                and event.get("payload", {}).get("type") == "patch_apply_end"
            )
            agent_messages = [
                event["payload"]["message"]
                for event in events
                if event.get("type") == "event_msg"
                and event.get("payload", {}).get("type") == "agent_message"
            ]

            self.assertIn(f"*** Update File: {target}", call["input"])
            self.assertNotIn(f"*** Add File: {target}", call["input"])
            self.assertEqual(patch_end["changes"][str(target)]["type"], "update")
            self.assertTrue(any("Edited `app.py` (+1 -1)" in message for message in agent_messages))

    def test_cursor_websearch_maps_to_native_web_search_call(self) -> None:
        events = self.importer.build_events(
            "11111111-1111-1111-1111-111111111111",
            [
                self.importer.ExportEntry(kind="user", text="look it up"),
                self.importer.ExportEntry(
                    kind="tool_call",
                    tool_name="WebSearch",
                    tool_call_id="web-1",
                    args={"search_term": "Codex CLI resume format"},
                ),
                self.importer.ExportEntry(kind="assistant", text="found it"),
            ],
            "test",
            Path("/home/yjh/bypass"),
            1_700_000_000_000,
            self.importer.ThreadDefaults(cli_version="0.142.3"),
        )

        web_call = next(
            event["payload"]
            for event in events
            if event.get("type") == "response_item"
            and event.get("payload", {}).get("type") == "web_search_call"
        )
        web_end = next(
            event["payload"]
            for event in events
            if event.get("type") == "event_msg"
            and event.get("payload", {}).get("type") == "web_search_end"
        )

        self.assertEqual(web_call["action"]["type"], "search")
        self.assertEqual(web_call["action"]["query"], "Codex CLI resume format")
        self.assertEqual(web_end["query"], "Codex CLI resume format")

    def test_cursor_websearch_result_is_preserved_in_model_context(self) -> None:
        events = self.importer.build_events(
            "11111111-1111-1111-1111-111111111111",
            [
                self.importer.ExportEntry(kind="user", text="look it up"),
                self.importer.ExportEntry(
                    kind="tool_call",
                    tool_name="WebSearch",
                    tool_call_id="web-1",
                    args={"search_term": "Codex CLI resume format"},
                ),
                self.importer.ExportEntry(
                    kind="tool_result",
                    tool_call_id="web-1",
                    result="Result title\nhttps://example.com\nSnippet important for follow-up\n",
                ),
            ],
            "test",
            Path("/home/yjh/bypass"),
            1_700_000_000_000,
            self.importer.ThreadDefaults(cli_version="0.142.3"),
        )

        response_messages = [
            event["payload"]
            for event in events
            if event.get("type") == "response_item"
            and event.get("payload", {}).get("type") == "message"
            and event.get("payload", {}).get("role") == "assistant"
        ]
        agent_messages = [
            event["payload"]["message"]
            for event in events
            if event.get("type") == "event_msg"
            and event.get("payload", {}).get("type") == "agent_message"
        ]

        self.assertTrue(
            any(
                "[Cursor WebSearch result]" in content.get("text", "")
                and "Snippet important for follow-up" in content.get("text", "")
                for message in response_messages
                for content in message.get("content", [])
            )
        )
        self.assertTrue(any("  └ Result title" in message for message in agent_messages))
        self.assertTrue(any("    Snippet important for follow-up" in message for message in agent_messages))

    def test_cursor_await_readlints_and_semantic_search_map_to_codex_context(self) -> None:
        events = self.importer.build_events(
            "11111111-1111-1111-1111-111111111111",
            [
                self.importer.ExportEntry(kind="user", text="check it"),
                self.importer.ExportEntry(
                    kind="tool_call",
                    tool_name="Await",
                    tool_call_id="await-1",
                    args={"task_id": "123", "block_until_ms": 60000},
                ),
                self.importer.ExportEntry(kind="tool_result", tool_call_id="await-1", result="done"),
                self.importer.ExportEntry(
                    kind="tool_call",
                    tool_name="ReadLints",
                    tool_call_id="lint-1",
                    args={"paths": ["/home/yjh/bypass/app.ts"]},
                ),
                self.importer.ExportEntry(kind="tool_result", tool_call_id="lint-1", result="No diagnostics"),
                self.importer.ExportEntry(
                    kind="tool_call",
                    tool_name="SemanticSearch",
                    tool_call_id="semantic-1",
                    args={"query": "session resume storage", "num_results": 5},
                ),
                self.importer.ExportEntry(kind="tool_result", tool_call_id="semantic-1", result="lib.py:42 match"),
            ],
            "test",
            Path("/home/yjh/bypass"),
            1_700_000_000_000,
            self.importer.ThreadDefaults(cli_version="0.142.3"),
        )

        calls = [
            event["payload"]
            for event in events
            if event.get("type") == "response_item"
            and event.get("payload", {}).get("type") == "function_call"
        ]
        by_name = {call["name"]: self.importer.json.loads(call["arguments"]) for call in calls}
        agent_messages = [
            event["payload"]["message"]
            for event in events
            if event.get("type") == "event_msg"
            and event.get("payload", {}).get("type") == "agent_message"
        ]

        self.assertEqual(by_name["wait"]["ids"], ["123"])
        self.assertEqual(by_name["wait"]["timeout_ms"], 60000)
        self.assertEqual(by_name["mcp__omx_code_intel__lsp_diagnostics"]["file"], "/home/yjh/bypass/app.ts")
        self.assertEqual(by_name["semantic_search"]["query"], "session resume storage")
        self.assertEqual(by_name["semantic_search"]["limit"], 5)
        self.assertTrue(any(message.startswith("Waited for `123`") for message in agent_messages))
        self.assertTrue(any(message.startswith("Read diagnostics for `/home/yjh/bypass/app.ts`") for message in agent_messages))
        self.assertTrue(any(message.startswith("Semantic searched `session resume storage`") for message in agent_messages))

    def test_read_and_grep_preserve_cursor_output_prefixes_in_codex_context(self) -> None:
        events = self.importer.build_events(
            "11111111-1111-1111-1111-111111111111",
            [
                self.importer.ExportEntry(kind="user", text="inspect it"),
                self.importer.ExportEntry(
                    kind="tool_call",
                    tool_name="Read",
                    tool_call_id="read-1",
                    args={"path": "/home/yjh/bypass/app.py", "offset": 40, "limit": 2},
                ),
                self.importer.ExportEntry(
                    kind="tool_result",
                    tool_call_id="read-1",
                    result="Exit code: 0\nOutput:\n  41 | def main():\n  42 |     return 1\n",
                ),
                self.importer.ExportEntry(
                    kind="tool_call",
                    tool_name="Grep",
                    tool_call_id="grep-1",
                    args={"pattern": "main", "path": "/home/yjh/bypass"},
                ),
                self.importer.ExportEntry(
                    kind="tool_result",
                    tool_call_id="grep-1",
                    result="Exit code: 0\nOutput:\napp.py:41:def main():\n",
                ),
            ],
            "test",
            Path("/home/yjh/bypass"),
            1_700_000_000_000,
            self.importer.ThreadDefaults(cli_version="0.142.3"),
        )

        calls = [
            event["payload"]
            for event in events
            if event.get("type") == "response_item"
            and event.get("payload", {}).get("type") == "function_call"
        ]
        outputs = [
            event["payload"]["output"]
            for event in events
            if event.get("type") == "response_item"
            and event.get("payload", {}).get("type") == "function_call_output"
        ]
        agent_messages = [
            event["payload"]["message"]
            for event in events
            if event.get("type") == "event_msg"
            and event.get("payload", {}).get("type") == "agent_message"
        ]

        commands = [self.importer.json.loads(call["arguments"])["cmd"] for call in calls]
        self.assertTrue(any(command.startswith("sed -n 41,42p") for command in commands))
        self.assertTrue(any(command.startswith("rg -n main") for command in commands))
        self.assertTrue(any("  41 | def main():" in output for output in outputs))
        self.assertTrue(any("app.py:41:def main():" in output for output in outputs))
        self.assertTrue(any("  └   41 | def main():" in message for message in agent_messages))
        self.assertTrue(any("  └ app.py:41:def main():" in message for message in agent_messages))

    def test_failed_patch_does_not_emit_successful_patch_apply_end_or_edited_replay(self) -> None:
        events = self.importer.build_events(
            "11111111-1111-1111-1111-111111111111",
            [
                self.importer.ExportEntry(kind="user", text="edit it"),
                self.importer.ExportEntry(
                    kind="tool_call",
                    tool_name="StrReplace",
                    tool_call_id="edit-1",
                    args={
                        "path": "/home/yjh/bypass/lib.py",
                        "old_string": "missing",
                        "new_string": "replacement",
                    },
                ),
                self.importer.ExportEntry(
                    kind="tool_result",
                    tool_call_id="edit-1",
                    result="Exit code: 1\nOutput:\nold_string not found\n",
                ),
            ],
            "test",
            Path("/home/yjh/bypass"),
            1_700_000_000_000,
            self.importer.ThreadDefaults(cli_version="0.142.3"),
        )

        patch_ends = [
            event
            for event in events
            if event.get("type") == "event_msg"
            and event.get("payload", {}).get("type") == "patch_apply_end"
        ]
        agent_messages = [
            event["payload"]["message"]
            for event in events
            if event.get("type") == "event_msg"
            and event.get("payload", {}).get("type") == "agent_message"
        ]
        outputs = [
            event["payload"]["output"]
            for event in events
            if event.get("type") == "response_item"
            and event.get("payload", {}).get("type") == "custom_tool_call_output"
        ]

        self.assertEqual(patch_ends, [])
        self.assertTrue(any(message.startswith("Patch failed") for message in agent_messages))
        self.assertFalse(any(message.startswith("Edited `lib.py`") for message in agent_messages))
        self.assertTrue(any("old_string not found" in output for output in outputs))

    def test_current_project_selection_prefers_exact_cwd_over_newer_parent(self) -> None:
        child_key = self.importer.cursor_project_key_for_path(Path("/tmp/c2c-selection/child"))
        parent_key = self.importer.cursor_project_key_for_path(Path("/tmp/c2c-selection"))
        selected, selected_by = self.importer.select_current_candidate(
            [
                self.importer.Candidate(
                    chat_id="parent",
                    updated_ms=2_000,
                    transcript_path=Path("/tmp/parent.jsonl"),
                    project_key=parent_key,
                ),
                self.importer.Candidate(
                    chat_id="child",
                    updated_ms=1_000,
                    transcript_path=Path("/tmp/child.jsonl"),
                    project_key=child_key,
                ),
            ],
            Path("/tmp/c2c-selection/child"),
        )

        self.assertEqual(selected.chat_id, "child")
        self.assertEqual(selected_by, "current-project")

    def test_current_project_selection_falls_back_to_nearest_parent(self) -> None:
        root_key = self.importer.cursor_project_key_for_path(Path("/tmp"))
        parent_key = self.importer.cursor_project_key_for_path(Path("/tmp/c2c-selection"))
        selected, _ = self.importer.select_current_candidate(
            [
                self.importer.Candidate(
                    chat_id="root",
                    updated_ms=3_000,
                    transcript_path=Path("/tmp/root.jsonl"),
                    project_key=root_key,
                ),
                self.importer.Candidate(
                    chat_id="parent",
                    updated_ms=1_000,
                    transcript_path=Path("/tmp/parent.jsonl"),
                    project_key=parent_key,
                ),
            ],
            Path("/tmp/c2c-selection/child"),
        )

        self.assertEqual(selected.chat_id, "parent")


if __name__ == "__main__":
    unittest.main()
