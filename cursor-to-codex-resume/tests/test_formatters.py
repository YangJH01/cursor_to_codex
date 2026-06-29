#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
