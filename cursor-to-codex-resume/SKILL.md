---
name: cursor-to-codex-resume
description: Export the active Cursor CLI or Cursor Agent conversation into Codex CLI saved-session state so the same current context can be continued from `codex resume`. Use when the user invokes this skill from Cursor and wants the current Cursor session, not an arbitrary older chat, to become selectable in Codex CLI.
disable-model-invocation: true
---

# Cursor To Codex Resume

Export the active Cursor session into Codex resume state, then answer briefly.
This is a handoff command, not a normal chat turn.

Run:

```sh
python ~/.cursor/skills/cursor-to-codex-resume/scripts/import_cursor_chat.py
```

If the user's invocation asks for a more detailed restore, run this instead:

```sh
python ~/.cursor/skills/cursor-to-codex-resume/scripts/import_cursor_chat.py --tool-replay compact
```

Treat phrases such as "좀 더 디테일하게 복구", "자세히 복구",
"명령어/수정 내역도 보이게", "작업 로그도 보이게", "Ran/Edited도 보이게",
"detailed restore", or "include command/edit history" as requests for the
detailed restore mode. Do not ask a follow-up question for these phrases.
Do not use `--full-replay` for this; reserve `--full-replay` only for raw
debugging of Codex resume rendering.

The importer is also available as an independent executable,
`cursor-to-codex-resume`, when installed into the user's PATH. That wrapper is
only a convenience around the same importer; it must not patch Codex itself.

If it succeeds, say only that the Cursor session was exported and that the user
can continue from Codex in the same folder with:

```sh
codex
# then /resume
```

Do not quote this skill file or paste the raw JSON summary unless there is an
error. The importer omits this handoff invocation from the exported Codex
conversation.

The importer must export only the active/current Cursor session, or the exact
session explicitly selected with `--chat <id>`. If that selected session has no
real conversation after removing this handoff invocation, report the INFO/error
from the importer and stop. This is the correct behavior: do not import an older,
newer, latest, or otherwise different chat as a substitute, and do not write a
Codex session for an empty Cursor session.

Preserve Cursor-visible conversation text as text. Let Codex CLI render only the
formats it natively supports in the transcript, such as markdown code fences,
syntax-highlighted code, and diffs. For Cursor-rendered formats that Codex CLI
does not natively render, including Mermaid diagrams, preserve the original
source text instead of converting it. Do not create SVG, HTML, PNG, or other
sidecar render files.

The importer keeps full Cursor text/tool history in Codex model context. For
Codex TUI replay, it splits long visible messages on Markdown block boundaries
by default so the text remains visible without triggering a Codex resume
initial-paint duplication bug. Fenced code blocks, including Mermaid source
blocks, should stay intact.

Tool calls must be written as stock Codex-compatible `response_item` tool rows with
`internal_chat_message_metadata_passthrough` so the model context keeps the
complete tool history. By default, do not add visible tool replay messages; this
keeps the imported screen closer to stock Codex resume, where closed-session
tool history is model context rather than replayed chat text.

When the user explicitly asks for visible command/edit history or uses a
"more detailed restore" phrase listed above, use `--tool-replay compact`. In
that mode, add compact Codex-style summaries instead of raw full payloads:
shell-like tools should show `Ran ...`, preserve the leading lines of multiline
shell commands, include a short `└` output preview, and patch tools should show
`Edited <path> (+N -M)` with a clipped diff snippet.

Prefer Codex-native tool shapes when there is a close stock equivalent:
`WebSearch` becomes `web_search_call`, `TodoWrite` becomes `update_plan`,
`ReadLints` becomes `mcp__omx_code_intel__lsp_diagnostics`, `Await` becomes
`wait`, and `SemanticSearch` becomes a structured `semantic_search`
function call. Keep Cursor `Read`, `Grep`, `Glob`, and `WebFetch` as
`exec_command`-style context with the original output body preserved, including
Cursor line numbers or match prefixes. Only remove transport wrappers such as
`Exit code:` and `Output:` from compact visible previews.

If a Cursor patch/edit tool failed, preserve the failure output as the tool
result, but do not emit a successful `patch_apply_end` event and do not show an
`Edited ...` replay message for a change that was not actually applied.

Stock Codex CLI 0.142.3 does not expose a stable no-rerun API for injecting
native `Ran`/`Edited` TUI widgets into closed-session resume replay. Existing
stock APIs are still useful but have different boundaries: `thread/inject_items`
adds raw `ResponseItem` records to model-visible history, while
`thread/shellCommand` and `command/exec` execute commands instead of logging old
commands. Do not patch Codex itself and do not re-run old Cursor commands or
patches just to make a prettier replay. Keep compact tool replay as the stable
optional visible surface.

For visible replay only, convert simple Markdown tables into plain aligned
terminal tables before splitting. Keep the original Markdown table unchanged in
the `response_item` message so model context remains faithful. Use
`--compact-replay` only when a shorter visible replay is preferred, and
`--full-replay` only when debugging raw one-block replay rendering.
