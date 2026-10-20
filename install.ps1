# Forge installer for Windows (PowerShell).
#
# Verified local install:
#   .\install.ps1 -Binary C:\path\to\forge.exe -NoModifyPath
#
# Remote release install (explicit repository required):
#   .\install.ps1 -Repository owner/repository

[CmdletBinding()]
param(
    [string]$Version = "",
    [string]$Binary = "",
    [string]$InstallDir = "",
    [string]$Repository = "",
    [string]$Token = "",
    [switch]$NoModifyPath,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
$App = 'forge'
$ConnectTimeoutSeconds = 10
$TotalTimeoutSeconds = 60
$ArchiveMaxBytes = 128MB
$ManifestMaxBytes = 1MB
$ApiMaxBytes = 1MB
$Repo = if (-not [string]::IsNullOrWhiteSpace($Repository)) {
    $Repository
} elseif (-not [string]::IsNullOrWhiteSpace($env:FORGE_RELEASE_REPO)) {
    $env:FORGE_RELEASE_REPO
} else {
    ''
}

function Write-Info($msg)    { Write-Host $msg }
function Write-Success($msg) { Write-Host $msg -ForegroundColor Green }
function Write-Warn($msg)    { Write-Host $msg -ForegroundColor Yellow }
function Write-Err($msg)     { Write-Host $msg -ForegroundColor Red }
function Write-Muted($msg)   { Write-Host $msg -ForegroundColor DarkGray }

function Show-Usage {
@"
Forge installer (Windows)

Usage: install.ps1 [options]

Options:
    -Help                    Show this help
    -Binary <path>           Install a local Forge binary
    -Repository <owner/repo> Download from an explicitly selected GitHub repository
    -Version <version>       Install a specific remote version (for example, 0.1.0)
    -InstallDir <path>       Override install location (default: %LOCALAPPDATA%\Programs\forge)
    -Token <token>           GitHub token for API/downloads (or GITHUB_TOKEN / GH_TOKEN)
    -NoModifyPath            Do not add the install directory to user PATH

Examples:
    .\install.ps1 -Binary C:\path\to\forge.exe -NoModifyPath
    .\install.ps1 -Repository owner/repository
    `$env:FORGE_RELEASE_REPO = 'owner/repository'; .\install.ps1 -Version 0.1.0
"@
}

if ($Help) { Show-Usage; exit 0 }

if ([string]::IsNullOrEmpty($Token)) {
    if (-not [string]::IsNullOrEmpty($env:GITHUB_TOKEN)) {
        $Token = $env:GITHUB_TOKEN
    } elseif (-not [string]::IsNullOrEmpty($env:GH_TOKEN)) {
        $Token = $env:GH_TOKEN
    }
}

function Get-Arch {
    $procArch = $env:PROCESSOR_ARCHITECTURE
    if ($null -eq $procArch) { $procArch = '' }
    switch ($procArch.ToLowerInvariant()) {
        'amd64' { return 'x86_64' }
        'x86'   { return 'x86_64' }
        'arm64' {
            Write-Warn 'ARM64 Windows release assets are not configured. Falling back to x86_64.'
            return 'x86_64'
        }
        default {
            Write-Warn "Unknown architecture '$procArch'. Defaulting to x86_64."
            return 'x86_64'
        }
    }
}

if ([string]::IsNullOrEmpty($InstallDir)) {
    if (-not [string]::IsNullOrEmpty($env:LOCALAPPDATA)) {
        $InstallDir = Join-Path $env:LOCALAPPDATA "Programs\forge"
    } else {
        $InstallDir = Join-Path $env:USERPROFILE '.local\bin'
    }
}
function Assert-RemoteRepository {
    if ([string]::IsNullOrWhiteSpace($script:Repo) -or $script:Repo -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$') {
        Write-Err 'Remote installation requires -Repository owner/repository or FORGE_RELEASE_REPO.'
        throw 'Remote repository is invalid.'
    }
}

function Normalize-Version([string]$candidate) {
    $normalized = $candidate -replace '^v', ''
    if ($normalized -notmatch '^[0-9]+\.[0-9]+\.[0-9]+([+.-][0-9A-Za-z+.-]+)?$') {
        throw 'Release version is invalid.'
    }
    return $normalized
}

function New-ForgeHttpClient {
    $handler = [System.Net.Http.HttpClientHandler]::new()
    $handler.AllowAutoRedirect = $false
    $client = [System.Net.Http.HttpClient]::new($handler)
    $client.DefaultRequestHeaders.UserAgent.ParseAdd('forge-installer')
    $client.DefaultRequestHeaders.Accept.ParseAdd('application/vnd.github+json')
    # Authorization is attached per request below; never forward a token to a
    # redirected release host such as objects.githubusercontent.com.
    return $client
}

function Invoke-CappedDownload([string]$Uri, [string]$Destination, [long]$MaxBytes, [string]$Label) {
    if ($Uri -notmatch '^https://') { throw 'Remote URL must use HTTPS.' }
    $deadline = [DateTime]::UtcNow.AddSeconds($TotalTimeoutSeconds)
    $client = New-ForgeHttpClient
    $response = $null
    $stream = $null
    $file = $null
    $failed = $false
    try {
        $current = [Uri]$Uri
        for ($redirects = 0; $redirects -le 5; $redirects++) {
            if ($current.Scheme -ne 'https') { throw 'Redirected URL must use HTTPS.' }
            $remaining = ($deadline - [DateTime]::UtcNow).TotalSeconds
            if ($remaining -le 0) { throw 'Remote request timed out.' }
            $requestTimeout = [Math]::Min($ConnectTimeoutSeconds, $remaining)
            $requestCts = [System.Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds($requestTimeout))
            $request = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Get, $current)
            if (-not [string]::IsNullOrEmpty($script:Token) -and
                ($current.Host -eq 'api.github.com' -or $current.Host -eq 'github.com')) {
                $request.Headers.Authorization = [System.Net.Http.Headers.AuthenticationHeaderValue]::new('Bearer', $script:Token)
            }
            try {
                $response = $client.SendAsync($request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead, $requestCts.Token).GetAwaiter().GetResult()
            } catch {
                throw 'Remote request failed.'
            } finally {
                $requestCts.Dispose()
                $request.Dispose()
            }
            $status = [int]$response.StatusCode
            if ($status -ge 300 -and $status -lt 400) {
                $location = $response.Headers.Location
                $response.Dispose(); $response = $null
                if ($null -eq $location -or $redirects -eq 5) { throw 'Remote redirect is invalid.' }
                $current = [Uri]::new($current, $location)
                continue
            }
            if (-not $response.IsSuccessStatusCode) { throw 'Remote request failed.' }
            break
        }
        if ($null -eq $response -or -not $response.IsSuccessStatusCode) { throw 'Remote request failed.' }
        $stream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $file = [System.IO.File]::Open($Destination, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        $buffer = New-Object byte[] 65536
        [long]$total = 0
        while ($true) {
            $remaining = ($deadline - [DateTime]::UtcNow).TotalSeconds
            if ($remaining -le 0) { throw 'Remote request timed out.' }
            $readCts = [System.Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds($remaining))
            try {
                $read = $stream.ReadAsync($buffer, 0, $buffer.Length, $readCts.Token).GetAwaiter().GetResult()
            } catch {
                throw 'Remote request timed out or failed.'
            } finally {
                $readCts.Dispose()
            }
            if ($read -le 0) { break }
            $total += $read
            if ($total -gt $MaxBytes) { throw "$Label exceeded its size limit." }
            $file.Write($buffer, 0, $read)
        }
        $file.Flush($true)
    } catch {
        $failed = $true
        throw [System.InvalidOperationException]::new($Label + ' download failed.')
    } finally {
        if ($null -ne $file) { $file.Dispose() }
        if ($null -ne $stream) { $stream.Dispose() }
        if ($null -ne $response) { $response.Dispose() }
        if ($null -ne $client) { $client.Dispose() }
        if ($failed) { Remove-Item -Force -LiteralPath $Destination -ErrorAction SilentlyContinue }
    }
}

function Resolve-Version {
    Assert-RemoteRepository
    if (-not [string]::IsNullOrEmpty($script:Version)) {
        try { return (Normalize-Version $script:Version) }
        catch { Write-Err 'Release version is invalid.'; throw [System.InvalidOperationException]::new('Release version is invalid.') }
    }
    $latestPath = Join-Path ([System.IO.Path]::GetTempPath()) ('.forge-latest-' + [System.Guid]::NewGuid().ToString('N'))
    try {
        Invoke-CappedDownload "https://api.github.com/repos/$script:Repo/releases/latest" $latestPath $ApiMaxBytes 'Latest-version response'
        $resp = Get-Content -Raw -LiteralPath $latestPath | ConvertFrom-Json
        if ($null -eq $resp.tag_name -or [string]::IsNullOrEmpty([string]$resp.tag_name)) { throw 'Missing release tag.' }
        return (Normalize-Version ([string]$resp.tag_name))
    } catch {
        Write-Err 'Failed to fetch or validate the latest version.'
        if ([string]::IsNullOrEmpty($script:Token)) {
            Write-Info 'Private releases or rate limits may require GITHUB_TOKEN, GH_TOKEN, or -Token.'
        }
        throw [System.InvalidOperationException]::new('Latest-version resolution failed.')
    } finally {
        Remove-Item -Force -LiteralPath $latestPath -ErrorAction SilentlyContinue
    }
}

function Get-ChecksumFromManifest([string]$Path, [string]$Archive) {
    $expected = $null
    $seen = @{}
    foreach ($line in [System.IO.File]::ReadAllLines($Path)) {
        if ([string]::IsNullOrEmpty($line) -or $line -notmatch '^([0-9A-Fa-f]{64})[ \t]{2}(\*?[^ \t]+)$') {
            throw 'Checksum manifest is malformed.'
        }
        $hash = $Matches[1]
        $name = $Matches[2].TrimStart('*')
        if ($seen.ContainsKey($name)) { throw 'Checksum manifest contains duplicate entries.' }
        $seen[$name] = $true
        if ($name -eq $Archive) {
            if ($null -ne $expected) { throw 'Checksum manifest contains duplicate archive entries.' }
            $expected = $hash
        }
    }
    if ($null -eq $expected) { throw 'Checksum manifest does not list the archive.' }
    return $expected
}

function Check-Existing($desiredVersion) {
    $existing = Get-Command $script:App -ErrorAction SilentlyContinue
    if ($null -eq $existing) { return }
    try {
        $vline = (& $script:App --version) 2>&1 | Select-Object -First 1
        $installed = ($vline -split '\s+')[-1]
    } catch {
        $installed = 'unknown'
    }
    if ($installed -eq $desiredVersion) {
        Write-Muted "Version $desiredVersion already installed at $($existing.Source)"
        exit 0
    }
    Write-Muted "Found existing forge at $($existing.Source) (v$installed); installing v$desiredVersion"
}

function Install-Binary($source) {
    if (-not (Test-Path -LiteralPath $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }
    $target = Join-Path $InstallDir 'forge.exe'
    $stale = "$target.old"
    $staged = Join-Path $InstallDir ('.forge-install-' + [System.Guid]::NewGuid().ToString('N'))
    $movedExisting = $false
    try {
        Copy-Item -Force -LiteralPath $source -Destination $staged -ErrorAction Stop
        if (Test-Path -LiteralPath $target) {
            if (Test-Path -LiteralPath $stale) {
                Remove-Item -Force -LiteralPath $stale -ErrorAction Stop
            }
            Move-Item -Force -LiteralPath $target -Destination $stale -ErrorAction Stop
            $movedExisting = $true
        }
        Move-Item -Force -LiteralPath $staged -Destination $target -ErrorAction Stop
        if (Test-Path -LiteralPath $stale) {
            Remove-Item -Force -LiteralPath $stale -ErrorAction SilentlyContinue
        }
        Write-Success "Installed: $target"
    } catch {
        if ($movedExisting -and -not (Test-Path -LiteralPath $target)) {
            try {
                Move-Item -Force -LiteralPath $stale -Destination $target -ErrorAction Stop
            } catch {
                Write-Err "Restore the previous binary from $stale."
            }
        }
        throw
    } finally {
        if (Test-Path -LiteralPath $staged) {
            Remove-Item -Force -LiteralPath $staged -ErrorAction SilentlyContinue
        }
    }
}

function Install-FromBinary {
    if (-not (Test-Path -LiteralPath $script:Binary -PathType Leaf)) {
        Write-Err "Binary not found at $script:Binary"
        exit 1
    }
    Write-Info "Installing Forge from $script:Binary"
    Install-Binary $script:Binary
}

function Download-And-Install($desiredVersion, $arch) {
    Assert-RemoteRepository
    $archive = "forge-windows-$arch.zip"
    $url = "https://github.com/$script:Repo/releases/download/v$desiredVersion/$archive"
    $sumsUrl = "https://github.com/$script:Repo/releases/download/v$desiredVersion/SHA256SUMS"
    $tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("forge-install-" + [System.Guid]::NewGuid().ToString('N'))
    $zipPath = Join-Path $tmpRoot $archive
    $sumsPath = Join-Path $tmpRoot 'SHA256SUMS'
    $extractDir = Join-Path $tmpRoot 'extract'
    try {
        New-Item -ItemType Directory -Path $extractDir -Force | Out-Null
        Write-Info "Installing Forge v$desiredVersion (windows-$arch)"
        Write-Muted 'Downloading release asset (HTTPS, bounded).'
        Invoke-CappedDownload $url $zipPath $ArchiveMaxBytes 'Release archive'
        Invoke-CappedDownload $sumsUrl $sumsPath $ManifestMaxBytes 'Checksum manifest'

        $expected = Get-ChecksumFromManifest $sumsPath $archive
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash
        if (-not ($actual -ieq $expected)) { throw 'Checksum verification failed.' }
        Write-Muted 'Checksum verified.'

        try {
            Expand-Archive -LiteralPath $zipPath -DestinationPath $extractDir -Force
        } catch {
            throw 'Archive extraction failed.'
        }
        $extractedExe = Join-Path $extractDir 'forge.exe'
        if (-not (Test-Path -LiteralPath $extractedExe -PathType Leaf)) {
            throw 'Archive did not contain the expected binary.'
        }
        Install-Binary $extractedExe
    } catch {
        Write-Err 'Remote installation failed.'
        throw [System.InvalidOperationException]::new('Remote installation failed.')
    } finally {
        # Covers failed download, checksum, extraction, and Install-Binary paths.
        Remove-Item -Recurse -Force -LiteralPath $tmpRoot -ErrorAction SilentlyContinue
    }
}

function Add-ToUserPath {
    if ($NoModifyPath) { return }
    $current = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($null -eq $current) { $current = '' }
    $paths = $current -split ';' | Where-Object { $_ -ne '' }
    foreach ($path in $paths) {
        if ($path.TrimEnd('\') -ieq $InstallDir.TrimEnd('\')) {
            Write-Muted "Install directory already on user PATH: $InstallDir"
            return
        }
    }
    $newPath = if ([string]::IsNullOrEmpty($current)) { $InstallDir } else { $InstallDir + ';' + $current }
    [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
    $env:Path = $InstallDir + ';' + $env:Path
    Write-Success "Added $InstallDir to user PATH"
}

function Write-GitHubPath {
    if ($env:GITHUB_ACTIONS -eq 'true' -and -not [string]::IsNullOrEmpty($env:GITHUB_PATH)) {
        Add-Content -Path $env:GITHUB_PATH -Value $InstallDir
        Write-Info "Added $InstallDir to `$GITHUB_PATH"
    }
}

if (-not [string]::IsNullOrEmpty($Binary)) {
    Install-FromBinary
} else {
    try {
        $arch = Get-Arch
        $desiredVersion = Resolve-Version
        Check-Existing $desiredVersion
        Download-And-Install $desiredVersion $arch
    } catch {
        exit 1
    }
}

Add-ToUserPath
Write-GitHubPath

Write-Host ''
Write-Success 'Forge is installed!'
Write-Muted 'Quickstart:'
Write-Success '  forge'
Write-Success '  forge -p "..."'
if (-not [string]::IsNullOrEmpty($Repo)) {
    Write-Muted "Release repository: https://github.com/$Repo"
}
