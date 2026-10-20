# Forge installation

Forge currently supports local builds and local binary installation. Remote
release downloads require an explicit repository because this fork has no
authorized public release location.

## Build from source

```sh
cd src-rust
cargo build --release --package claurst --bin forge
./target/release/forge --version
```

The Cargo package retains the internal `claurst` crate name for compatibility.
The shipped executable is `forge` (`forge.exe` on Windows).

## Install a local binary

Linux and macOS:

```sh
./install.sh --binary ./src-rust/target/release/forge --no-modify-path
forge --version
```

Windows PowerShell:

```powershell
.\install.ps1 -Binary .\src-rust\target\release\forge.exe -NoModifyPath
forge --version
```

Both installers accept a custom install directory. The Unix default is
`~/.local/bin`. The Windows default is
`%LOCALAPPDATA%\Programs\forge`.

## Private or future release repositories

Set the repository explicitly before a remote install:

```sh
FORGE_RELEASE_REPO=owner/repository ./install.sh
```

```powershell
.\install.ps1 -Repository owner/repository
```

Remote assets must be named `forge-<platform>-<architecture>` with the archive
extensions used by the installers. Each release must include `SHA256SUMS`.
Neither installer falls back to an upstream project.

## Upgrade an existing Forge install

The upgrade command uses the same explicit repository requirement as the installers.
Set `FORGE_RELEASE_REPO` or pass `--repository owner/repository` before a remote
upgrade. The selected release must publish the target Forge archive and an exact
`SHA256SUMS` entry.

```sh
FORGE_RELEASE_REPO=owner/repository forge upgrade
forge upgrade --repository owner/repository --version 0.1.0
```

## Local npm wrapper

The private npm package copies an existing binary. It does not publish or
download unless `FORGE_RELEASE_REPO` is set explicitly.

```sh
cd npm
FORGE_BINARY_PATH=/absolute/path/to/forge npm install -g .
forge --version
```

## VS Code extension

The extension in `editors/vscode` starts `forge acp`. Compile and test it
locally:

```sh
cd editors/vscode
npm run compile
npm test
```

Set `forge.executablePath` in VS Code if the executable is not on `PATH`.

## Verify and remove

```sh
forge --version
forge --help
```

Remove the installed executable from the selected install directory. Forge
configuration and state live under `FORGE_HOME` when set, otherwise the normal
platform configuration directory for Forge. Remove that data separately only
when it is no longer needed.
