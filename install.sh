#!/usr/bin/env bash
# Forge installer for Linux, macOS, and Windows (Git Bash / MSYS / Cygwin).
#
# Verified local usage:
#   ./install.sh --binary /absolute/path/to/forge --no-modify-path

set -euo pipefail

APP=forge
REPO=${FORGE_RELEASE_REPO:-}

# ANSI colours
MUTED='\033[0;2m'
RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[38;5;214m'
NC='\033[0m'

usage() {
    cat <<EOF
Forge installer

Usage: install.sh [options]

Options:
    -h, --help              Display this help message
    -v, --version <version> Install a specific version (e.g., 0.1.0 or v0.1.0)
    -b, --binary <path>     Install from a local binary instead of downloading
        --repository <repo> Explicit release repository (owner/name)
        --token <token>     GitHub token for API/downloads (or set GITHUB_TOKEN / GH_TOKEN)
        --no-modify-path    Don't modify shell config files (.zshrc, .bashrc, etc.)
        --install-dir <dir> Override install location
                            (default: ${XDG_BIN_HOME:-~/.local/bin}; on Windows:
                            %LOCALAPPDATA%\\Programs\\forge)

Examples:
    ./install.sh --binary /path/to/forge --no-modify-path
    FORGE_RELEASE_REPO=owner/repository ./install.sh --version 0.1.0
    GITHUB_TOKEN=ghp_... ./install.sh
EOF
}

print_message() {
    local level=$1
    local message=$2
    local color=""
    case "$level" in
        info)    color="$NC" ;;
        success) color="$GREEN" ;;
        warning) color="$ORANGE" ;;
        error)   color="$RED" ;;
        *)       color="$NC" ;;
    esac
    printf "${color}%b${NC}\n" "$message"
}

requested_version=${VERSION:-}
no_modify_path=false
binary_path=""
install_dir_override=""
# Prefer explicit --token; otherwise GITHUB_TOKEN, then GH_TOKEN (gh CLI convention).
github_token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

# Remote-install safety limits. Local --binary installs never invoke these paths.
CONNECT_TIMEOUT_SECONDS=10
TOTAL_TIMEOUT_SECONDS=60
ARCHIVE_MAX_BYTES=$((128 * 1024 * 1024))
MANIFEST_MAX_BYTES=$((1 * 1024 * 1024))
API_MAX_BYTES=$((1 * 1024 * 1024))

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--version)
            if [[ -n "${2:-}" ]]; then
                requested_version="$2"
                shift 2
            else
                print_message error "Error: --version requires a version argument"
                exit 1
            fi
            ;;
        -b|--binary)
            if [[ -n "${2:-}" ]]; then
                binary_path="$2"
                shift 2
            else
                print_message error "Error: --binary requires a path argument"
                exit 1
            fi
            ;;
        --token)
            if [[ -n "${2:-}" ]]; then
                github_token="$2"
                shift 2
            else
                print_message error "Error: --token requires a token argument"
                exit 1
            fi
            ;;
        --repository)
            if [[ -n "${2:-}" ]]; then
                REPO="$2"
                shift 2
            else
                print_message error "Error: --repository requires owner/name"
                exit 1
            fi
            ;;
        --no-modify-path)
            no_modify_path=true
            shift
            ;;
        --install-dir)
            if [[ -n "${2:-}" ]]; then
                install_dir_override="$2"
                shift 2
            else
                print_message error "Error: --install-dir requires a path argument"
                exit 1
            fi
            ;;
        *)
            print_message warning "Warning: Unknown option '$1'"
            shift
            ;;
    esac
done

# True when running under Git Bash / MSYS / Cygwin on Windows.
is_windows_shell() {
    case "$(uname -s 2>/dev/null || echo unknown)" in
        MINGW*|MSYS*|CYGWIN*) return 0 ;;
        *) return 1 ;;
    esac
}

# Convert a Windows path (C:\foo or C:/foo) to a Unix path when cygpath exists.
to_unix_path() {
    local p="$1"
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -u "$p"
    else
        # MSYS/Git Bash often accept mixed paths after normalizing slashes.
        echo "${p//\\//}"
    fi
}

# Default install dir matches install.ps1 on Windows:
#   %LOCALAPPDATA%\Programs\forge  (fallback: %USERPROFILE%\.local\bin)
# Unix: ${XDG_BIN_HOME:-~/.local/bin}
default_install_dir() {
    if is_windows_shell; then
        if [[ -n "${LOCALAPPDATA:-}" ]]; then
            echo "$(to_unix_path "$LOCALAPPDATA")/Programs/forge"
        elif [[ -n "${USERPROFILE:-}" ]]; then
            echo "$(to_unix_path "$USERPROFILE")/.local/bin"
        else
            echo "${HOME}/.local/bin"
        fi
    else
        echo "${XDG_BIN_HOME:-$HOME/.local/bin}"
    fi
}

INSTALL_DIR="${install_dir_override:-$(default_install_dir)}"

# ----- Detect platform & arch -----
detect_target() {
    local raw_os arch
    raw_os=$(uname -s)
    case "$raw_os" in
        Darwin*)              os="macos" ;;
        Linux*)               os="linux" ;;
        MINGW*|MSYS*|CYGWIN*) os="windows" ;;
        *)
            print_message error "Unsupported OS: $raw_os"
            exit 1
            ;;
    esac

    arch=$(uname -m)
    case "$arch" in
        x86_64|amd64) arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        *)
            print_message error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac

    # Apple Silicon under Rosetta: prefer the native arm64 binary.
    if [ "$os" = "macos" ] && [ "$arch" = "x86_64" ]; then
        rosetta_flag=$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)
        if [ "$rosetta_flag" = "1" ]; then
            arch="aarch64"
        fi
    fi

    # Windows releases currently ship x86_64 only (same as install.ps1).
    if [ "$os" = "windows" ] && [ "$arch" != "x86_64" ]; then
        print_message warning "Windows ${arch} is not currently supported in releases. Falling back to x86_64."
        arch="x86_64"
    fi

    target="${os}-${arch}"
    if [ "$os" = "windows" ]; then
        archive_ext="zip"
        binary_name="${APP}.exe"
    else
        archive_ext="tar.gz"
        binary_name="$APP"
    fi
}

# ----- Pre-flight: required tools -----
check_required_tools() {
    local missing=()
    if [[ -z "$binary_path" ]]; then
        command -v curl >/dev/null 2>&1 || missing+=("curl")
        command -v head >/dev/null 2>&1 || missing+=("head")
        command -v wc >/dev/null 2>&1 || missing+=("wc")
        if is_windows_shell; then
            command -v unzip >/dev/null 2>&1 || missing+=("unzip")
        else
            command -v tar >/dev/null 2>&1 || missing+=("tar")
        fi
    fi
    if [ "${#missing[@]}" -gt 0 ]; then
        print_message error "Missing required tools: ${missing[*]}"
        print_message info "Please install them and try again."
        exit 1
    fi
}

# Download through a bounded reader.  The response body is written to a part
# file and only CAP+1 bytes can be persisted. Redirects are handled one hop at
# a time instead of using -L: curl's manual warns that custom -H headers are
# sent on every redirected request, including other hosts. A token is attached
# only when the current host is a GitHub API/release host.
download_capped() {
    local url="$1" output="$2" cap="$3" label="$4"
    local part="${output}.part" headers="${output}.headers"
    local current="$url" redirects=0 started=$SECONDS
    rm -f -- "$part" "$headers"
    while true; do
        local elapsed=$((SECONDS - started))
        local remaining=$((TOTAL_TIMEOUT_SECONDS - elapsed))
        if (( remaining <= 0 )); then
            rm -f -- "$part" "$headers"
            print_message error "${label} download timed out."
            return 1
        fi
        local request_timeout="$CONNECT_TIMEOUT_SECONDS"
        if (( request_timeout > remaining )); then request_timeout="$remaining"; fi
        local host="${current#https://}"
        host="${host%%/*}"
        local -a args=(-fsS --proto '=https' --proto-redir '=https' --max-redirs 0 \
            --connect-timeout "$request_timeout" --max-time "$remaining" \
            -D "$headers" -H "User-Agent: forge-installer" \
            -H "Accept: application/vnd.github+json")
        if [[ -n "$github_token" && ( "$host" == "api.github.com" || "$host" == "github.com" ) ]]; then
            args+=(-H "Authorization: Bearer ${github_token}")
        fi

        # Do not print curl diagnostics: they can include request URLs or proxy data.
        set +e
        curl "${args[@]}" "$current" 2>/dev/null | head -c "$((cap + 1))" > "$part" 2>/dev/null
        local -a statuses=("${PIPESTATUS[@]}")
        set -e
        local curl_status="${statuses[0]:-1}" status_code location size
        # A proxy may leave CONNECT and 100-Continue status lines in the
        # header dump. Select the final HTTP status line, not the first one.
        status_code=$(awk '/^HTTP\/[0-9.]+[[:space:]]+[0-9][0-9][0-9]([[:space:]]|$)/ { code=$2 } END { gsub(/\r/, "", code); print code }' "$headers")
        if [[ "$status_code" =~ ^3[0-9][0-9]$ ]]; then
            location=$(sed -n 's/^[Ll]ocation:[[:space:]]*//p' "$headers" | tail -n 1 | tr -d '\r')
            rm -f -- "$part" "$headers"
            if [[ -z "$location" || ! "$location" =~ ^https:// ]]; then
                print_message error "${label} redirect was not HTTPS."
                return 1
            fi
            redirects=$((redirects + 1))
            if (( redirects > 5 )); then
                print_message error "${label} exceeded its redirect limit."
                return 1
            fi
            current="$location"
            continue
        fi
        size=$(wc -c < "$part" | tr -d '[:space:]')
        if [[ ! "$size" =~ ^[0-9]+$ ]] || (( size > cap )); then
            rm -f -- "$part" "$headers"
            print_message error "${label} exceeded its size limit."
            return 1
        fi
        if (( curl_status != 0 )); then
            rm -f -- "$part" "$headers"
            print_message error "${label} download failed."
            return 1
        fi
        rm -f -- "$headers"
        mv -f -- "$part" "$output"
        return 0
    done
}

validate_version() {
    local value="$1"
    if [[ ! "$value" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+.-][0-9A-Za-z+.-]+)?$ ]]; then
        print_message error "Invalid release version."
        exit 1
    fi
}

validate_remote_inputs() {
    if [[ ! "$REPO" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
        print_message error "Remote install requires --repository owner/name or FORGE_RELEASE_REPO."
        print_message info "For the verified local path, use --binary /absolute/path/to/forge."
        exit 1
    fi
    if [[ -n "$requested_version" ]]; then
        requested_version="${requested_version#v}"
        validate_version "$requested_version"
    fi
}

resolve_version() {
    # Validate all user-controlled URL components before any remote request.
    validate_remote_inputs
    if [[ -n "$requested_version" ]]; then
        specific_version="$requested_version"
        return
    fi

    local latest_file
    latest_file=$(mktemp "${TMPDIR:-/tmp}/forge-latest.XXXXXX")
    if ! download_capped \
        "https://api.github.com/repos/${REPO}/releases/latest" \
        "$latest_file" "$API_MAX_BYTES" "Latest-version response"; then
        rm -f "$latest_file"
        print_message error "Failed to fetch the latest version."
        if [[ -z "$github_token" ]]; then
            print_message info "Private releases or rate limits may require GITHUB_TOKEN, GH_TOKEN, or --token."
        fi
        exit 1
    fi
    # The response is capped before parsing. Extract one tag and validate it as
    # a release version before constructing the release-asset URL.
    specific_version=$(sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\(v\{0,1\}[0-9][^"]*\)".*/\1/p' "$latest_file" | head -n 1)
    rm -f "$latest_file"
    specific_version="${specific_version#v}"
    if [[ -z "$specific_version" ]]; then
        print_message error "Latest-version response did not contain a release tag."
        exit 1
    fi
    validate_version "$specific_version"
}

verify_checksum_manifest() {
    local manifest="$1" archive_name="$2"
    local expected
    # Every non-empty line must be a strict sha256sum record. Track every name
    # to reject duplicate records, then require exactly one exact archive entry.
    if ! expected=$(awk -v wanted="$archive_name" '
        BEGIN { bad=0; matches=0 }
        {
            if (NF != 2 || length($1) != 64 || $1 !~ /^[0-9A-Fa-f]+$/) { bad=1; next }
            name=$2; sub(/^\*/, "", name)
            if (seen[name]++) { bad=1 }
            if (name == wanted) { matches++; expected=$1 }
        }
        END {
            if (bad || matches != 1) { exit 1 }
            print expected
        }
    ' "$manifest"); then
        print_message error "Checksum manifest is malformed, duplicated, or missing the archive."
        return 1
    fi
    printf '%s\n' "$expected"
}

install_tmp_dir=""
cleanup_install() {
    if [[ -n "${install_tmp_dir:-}" && -d "$install_tmp_dir" ]]; then
        rm -rf -- "$install_tmp_dir"
    fi
}

# ----- Already-installed check -----
check_existing_install() {
    if command -v "$APP" >/dev/null 2>&1; then
        existing_path=$(command -v "$APP")
        installed_version=$("$APP" --version 2>/dev/null | awk '{print $NF}' || echo "unknown")
        if [[ "$installed_version" == "$specific_version" ]]; then
            print_message info "${MUTED}Version ${NC}$specific_version${MUTED} already installed at ${NC}$existing_path"
            print_message info "${MUTED}Use --version to install a different version, or pass a different one to upgrade.${NC}"
            exit 0
        fi
        print_message info "${MUTED}Found existing forge at ${NC}$existing_path${MUTED} (v$installed_version) - upgrading to v$specific_version${NC}"
    fi
}

# ----- Download & extract -----
download_and_install() {
    local archive="${APP}-${target}.${archive_ext}"
    local url="https://github.com/${REPO}/releases/download/v${specific_version}/${archive}"
    local sums_url="https://github.com/${REPO}/releases/download/v${specific_version}/SHA256SUMS"
    local tmp_dir
    # mktemp -t is GNU-style; fall back for older / BSD layouts (and Git Bash).
    tmp_dir=$(mktemp -d -t forge-install-XXXXXX 2>/dev/null \
        || mktemp -d "${TMPDIR:-/tmp}/forge-install.XXXXXX")
    install_tmp_dir="$tmp_dir"
    trap cleanup_install EXIT

    print_message info "${MUTED}Installing ${NC}${APP} ${MUTED}v${NC}${specific_version} ${MUTED}(${target})${NC}"
    print_message info "${MUTED}Downloading release asset (HTTPS, bounded).${NC}"
    if ! download_capped "$url" "$tmp_dir/$archive" "$ARCHIVE_MAX_BYTES" "Release archive"; then
        print_message error "Release archive download failed."
        if [[ -z "$github_token" ]]; then
            print_message info "Private releases require GITHUB_TOKEN / GH_TOKEN or --token."
        fi
        exit 1
    fi

    # The manifest is independently bounded and must contain one unique,
    # well-formed entry for this exact archive before extraction.
    if ! download_capped "$sums_url" "$tmp_dir/SHA256SUMS" "$MANIFEST_MAX_BYTES" "Checksum manifest"; then
        print_message error "Could not fetch the required checksum manifest."
        exit 1
    fi
    local expected actual expected_lower actual_lower
    if ! expected=$(verify_checksum_manifest "$tmp_dir/SHA256SUMS" "$archive"); then
        exit 1
    fi
    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$tmp_dir/$archive" | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
        actual=$(shasum -a 256 "$tmp_dir/$archive" | awk '{print $1}')
    else
        print_message error "Checksum verification requires sha256sum or shasum."
        exit 1
    fi
    expected_lower=$(printf '%s' "$expected" | tr '[:upper:]' '[:lower:]')
    actual_lower=$(printf '%s' "$actual" | tr '[:upper:]' '[:lower:]')
    if [[ "$actual_lower" != "$expected_lower" ]]; then
        print_message error "Checksum verification failed for the release archive."
        exit 1
    fi
    print_message info "${MUTED}Checksum verified.${NC}"

    print_message info "${MUTED}Extracting...${NC}"
    if [[ "$archive_ext" == "zip" ]]; then
        if ! unzip -o -q "$tmp_dir/$archive" -d "$tmp_dir"; then
            print_message error "Extract failed."
            exit 1
        fi
    else
        if ! tar -xzf "$tmp_dir/$archive" -C "$tmp_dir"; then
            print_message error "Extract failed."
            exit 1
        fi
    fi

    if [[ ! -f "$tmp_dir/$binary_name" ]]; then
        print_message error "Archive did not contain the expected binary."
        exit 1
    fi

    install_binary "$tmp_dir/$binary_name"
}

install_from_binary() {
    if [[ ! -f "$binary_path" ]]; then
        print_message error "Binary not found at $binary_path"
        exit 1
    fi
    print_message info "${MUTED}Installing ${NC}${APP} ${MUTED}from ${NC}$binary_path"
    install_binary "$binary_path"
}

install_binary() {
    local source="$1"
    # Prefer binary_name from detect_target; local --binary installs may not set it.
    local dest_name="${binary_name:-$APP}"
    if is_windows_shell && [[ "$dest_name" != *.exe ]]; then
        dest_name="${dest_name}.exe"
    fi
    local target_path="${INSTALL_DIR}/${dest_name}"
    local staged
    mkdir -p -- "$INSTALL_DIR"
    staged=$(mktemp "${INSTALL_DIR}/.forge-install.XXXXXX")
    if ! cp "$source" "$staged" || ! chmod 755 "$staged"; then
        rm -f "$staged"
        print_message error "Could not stage the installation binary."
        return 1
    fi

    # Windows holds an exclusive lock on a running .exe — rename the old binary
    # aside first, then remove it after the new binary is in place (same as
    # install.ps1). The lock is on the open handle, not the new path name.
    local stale="${target_path}.old"
    if is_windows_shell && [[ -f "$target_path" ]]; then
        if ! rm -f "$stale" || ! mv -f "$target_path" "$stale"; then
            rm -f "$staged"
            print_message error "Could not move the installed binary."
            return 1
        fi
    fi

    if ! mv -f "$staged" "$target_path"; then
        if [[ -f "$stale" && ! -e "$target_path" ]]; then
            if ! mv -f "$stale" "$target_path"; then
                print_message error "Restore the previous binary from $stale."
            fi
        fi
        rm -f "$staged"
        print_message error "Could not install the staged binary."
        return 1
    fi

    if [[ -f "$stale" ]]; then
        rm -f "$stale" 2>/dev/null || true
    fi

    # On macOS, strip the quarantine attribute so Gatekeeper doesn't block the unsigned binary.
    if [[ "$(uname -s)" == "Darwin" ]]; then
        xattr -dr com.apple.quarantine "$target_path" 2>/dev/null || true
    fi

    print_message success "Installed: $target_path"
}

# ----- PATH modification -----
# On Windows, match install.ps1: update the user PATH env var (inherited by
# PowerShell, cmd, and Git Bash). On Unix, append to shell config files.
add_to_windows_user_path() {
    local win_dir
    if command -v cygpath >/dev/null 2>&1; then
        win_dir=$(cygpath -w "$INSTALL_DIR")
    else
        # Best-effort: turn /c/Users/... into C:\Users\...
        win_dir="$INSTALL_DIR"
        if [[ "$win_dir" =~ ^/([a-zA-Z])/(.*) ]]; then
            win_dir="${BASH_REMATCH[1]^}:\\${BASH_REMATCH[2]//\//\\}"
        else
            win_dir="${win_dir//\//\\}"
        fi
    fi

    # Use PowerShell so we only touch the User scope (same as install.ps1).
    local ps_script
    ps_script=$(cat <<'PSEOF'
$ErrorActionPreference = 'Stop'
$installDir = $env:FORGE_INSTALL_DIR_WIN
$current = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($null -eq $current) { $current = '' }
$paths = $current -split ';' | Where-Object { $_ -ne '' }
foreach ($p in $paths) {
    if ($p.TrimEnd('\') -ieq $installDir.TrimEnd('\')) {
        Write-Output 'already'
        exit 0
    }
}
if ([string]::IsNullOrEmpty($current)) {
    $newPath = $installDir
} else {
    $newPath = $installDir + ';' + $current
}
[Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
Write-Output 'added'
PSEOF
)

    local result
    if ! result=$(FORGE_INSTALL_DIR_WIN="$win_dir" powershell.exe -NoProfile -NonInteractive -Command "$ps_script" 2>/dev/null); then
        print_message warning "Could not update Windows user PATH. Add manually: $win_dir"
        return
    fi
    result=$(echo "$result" | tr -d '\r' | tail -n 1)
    # Visible in this Git Bash session immediately.
    export PATH="$INSTALL_DIR:$PATH"
    if [[ "$result" == "already" ]]; then
        print_message info "${MUTED}Install dir already on user PATH: ${NC}$win_dir"
    else
        print_message success "Added $win_dir to user PATH"
        print_message info "${MUTED}Open a new terminal for the change to take effect everywhere.${NC}"
    fi
}

add_to_unix_path() {
    if [[ ":$PATH:" == *":$INSTALL_DIR:"* ]]; then
        return  # already on PATH for this session
    fi

    XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-$HOME/.config}
    local current_shell config_files=""
    current_shell=$(basename "${SHELL:-/bin/bash}")

    case "$current_shell" in
        fish)
            config_files="$HOME/.config/fish/config.fish"
            ;;
        zsh)
            config_files="${ZDOTDIR:-$HOME}/.zshrc ${ZDOTDIR:-$HOME}/.zshenv $XDG_CONFIG_HOME/zsh/.zshrc"
            ;;
        bash)
            config_files="$HOME/.bashrc $HOME/.bash_profile $HOME/.profile"
            ;;
        ash|sh)
            config_files="$HOME/.profile /etc/profile"
            ;;
        *)
            config_files="$HOME/.bashrc $HOME/.bash_profile $HOME/.profile"
            ;;
    esac

    local config_file=""
    for file in $config_files; do
        if [[ -f "$file" ]]; then
            config_file="$file"
            break
        fi
    done

    # Fall back to creating ~/.profile if nothing exists.
    if [[ -z "$config_file" ]]; then
        config_file="$HOME/.profile"
        touch "$config_file"
    fi

    local path_line
    if [[ "$current_shell" == "fish" ]]; then
        path_line="fish_add_path $INSTALL_DIR"
    else
        path_line="export PATH=\"$INSTALL_DIR:\$PATH\""
    fi

    if grep -Fq "$INSTALL_DIR" "$config_file" 2>/dev/null; then
        print_message info "${MUTED}PATH entry already in ${NC}$config_file"
        return
    fi

    {
        echo ""
        echo "# forge"
        echo "$path_line"
    } >> "$config_file"
    print_message success "Added $INSTALL_DIR to PATH in $config_file"
    print_message info "${MUTED}Restart your shell or run:${NC} source $config_file"
}

add_to_path_if_needed() {
    if [[ "$no_modify_path" == "true" ]]; then
        return
    fi

    if is_windows_shell; then
        add_to_windows_user_path
    else
        add_to_unix_path
    fi
}

# ----- GitHub Actions environment hint -----
github_path_hint() {
    if [ -n "${GITHUB_ACTIONS-}" ] && [ "${GITHUB_ACTIONS}" = "true" ] && [ -n "${GITHUB_PATH-}" ]; then
        echo "$INSTALL_DIR" >> "$GITHUB_PATH"
        print_message info "Added $INSTALL_DIR to \$GITHUB_PATH"
    fi
}

# ----- Main flow -----
main() {
    check_required_tools

    if [[ -n "$binary_path" ]]; then
        # Local binary install - skip detection & version resolution.
        specific_version="local"
        install_from_binary
    else
        detect_target
        resolve_version
        check_existing_install
        download_and_install
    fi

    add_to_path_if_needed
    github_path_hint

    # Goodbye banner
    echo ""
    print_message success "forge is installed!"
    echo ""
    echo -e "${MUTED}Quickstart:${NC}"
    echo -e "  ${MUTED}# Set an API key${NC}"
    echo -e "  export ANTHROPIC_API_KEY=sk-ant-..."
    echo -e ""
    echo -e "  ${MUTED}# Open a new terminal, then:${NC}"
    echo -e "  ${GREEN}forge${NC}              ${MUTED}# Interactive TUI${NC}"
    echo -e "  ${GREEN}forge -p \"...\"${NC}       ${MUTED}# Headless one-shot${NC}"
    echo ""
    echo -e "${MUTED}Release repository: ${NC}${REPO:-not configured}"
}

main "$@"
