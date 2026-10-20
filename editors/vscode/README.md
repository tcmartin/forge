# Forge for VS Code

This local extension connects VS Code to the Forge coding agent. It starts
`forge acp` and communicates over the Agent Client Protocol (ACP).

## Requirements

- Build or install the `forge` executable.
- Keep `forge` on `PATH`, or set `forge.executablePath` to its absolute path.

## Commands

- **Forge: Open Chat** opens the chat panel and starts a session.
- **Forge: New Session** starts a fresh session.
- **Forge: Stop Current Turn** cancels the active turn.

The chat panel streams messages and tool updates. When Forge requests tool
permission, the extension displays the choices from the ACP request. Closing
that picker returns a cancelled outcome.

## Local development

```sh
npm run compile
npm test
```

This package is private and has no configured marketplace publisher or remote
release repository.
