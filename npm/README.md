# Forge local npm package

This private package exposes the `forge` command for a verified local Forge binary. It is not published to npm and does not assume a public release repository.

## Install from a local binary

```bash
FORGE_BINARY_PATH=/absolute/path/to/forge npm install -g ./npm
forge --version
```

The postinstall step replaces the installed binary with the supplied file and marks it executable on Unix. It records the package version, platform and binary checksum. Repeated installs verify those records before reusing a binary.

Installs with lifecycle scripts disabled require an explicit rebuild:

```bash
FORGE_BINARY_PATH=/absolute/path/to/forge npm rebuild forge-coding-agent-local
```

The launcher rejects missing records, a different package version, or a changed file identity. Normal launches use file metadata; installer reuse also verifies the full checksum. These checks detect stale installs and accidental changes. They do not authenticate files against modification by the same operating-system user.

## Install from an explicit release repository

```bash
FORGE_RELEASE_REPO=owner/repository npm install -g ./npm
```

The repository must contain a `v0.1.7` release with a platform archive named `forge-<platform>-<architecture>.tar.gz` or `.zip`. Forge never falls back to the upstream Claurst release channel.

The package retains the upstream GPL-3.0 license and attribution. Public npm publication remains disabled until a package scope and release repository are explicitly chosen.
