# Instructions and extensions

Forge loads project and global `AGENTS.md` instructions by default.
`--no-claude-md` disables these files. The saved
`config.disable_claude_mds` setting also disables them.

`--bare` disables instruction files, configured hooks and external command,
skill and plugin loading. Plugin management commands cannot reload extensions
into a bare session. Bundled commands remain available.

`--disable-slash-commands` sends slash text as ordinary input in the TUI.
An expanded custom command otherwise becomes one user turn. Its original slash
text is not appended as another turn.

Use `/plugin disable NAME` or `/plugin enable NAME` to save plugin state.
Run `/plugin reload` or `/reload-plugins` to apply it in the current session.
Reload updates the shared metadata used by listing and invocation. Added and
removed plugins take effect without restarting. Settings errors are reported;
a failed state update does not overwrite a malformed settings file.

Plugin commands retain their manifest capability checks. Project plugin MCP
servers require the existing project trust approval, including
`--trust-project-mcp` for explicit automation. Disabled plugins start no server.
Metadata reload does not activate dormant process-global hooks.

The current acceptance slice passes 19 instruction/plugin workflows and 12
saved-fork/output/TUI regressions on immutable debug `8a24e177`, source
`02267c5f`. Workspace check, strict all-target Clippy and 2,534 workspace tests
pass at that source. Evidence is retained in the parent workspace under
`.omx/state/forge/verification/u014-instructions-plugins/`.

U01.4 remains in progress. Two actual probes still fail: directory-based
`.agents/skills/NAME/SKILL.md` discovery and plugin-contributed skill invocation.
The remaining trust/reload audit and final-release acceptance are required.
