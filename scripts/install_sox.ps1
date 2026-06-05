param(
    [string]$Root = ""
)

$ErrorActionPreference = "Stop"

if (-not $Root) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$existing = Get-Command sox -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "SoX is already available: $($existing.Source)"
    exit 0
}

$toolRoot = Join-Path $Root ".tools\sox"
$downloadRoot = Join-Path $Root ".tools\downloads"
$archive = Join-Path $downloadRoot "sox-14.4.2-win32.zip"
$url = "https://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2-win32.zip?use_mirror=versaweb"

function Test-ZipSignature {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $false
    }
    $bytes = Get-Content -Path $Path -Encoding Byte -TotalCount 4
    return $bytes.Length -ge 4 -and $bytes[0] -eq 0x50 -and $bytes[1] -eq 0x4B
}

$repoSox = Get-ChildItem -Path $toolRoot -Recurse -Filter sox.exe -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($repoSox) {
    Write-Host "Repo-local SoX is already available: $($repoSox.FullName)"
    exit 0
}

New-Item -ItemType Directory -Force -Path $toolRoot, $downloadRoot | Out-Null

if ((Test-Path $archive) -and -not (Test-ZipSignature -Path $archive)) {
    Remove-Item -Force $archive
}

if (-not (Test-Path $archive)) {
    Write-Host "Downloading portable SoX for Windows..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source -L --fail --output $archive $url
    } else {
        Invoke-WebRequest -Uri $url -OutFile $archive
    }
}

if (-not (Test-ZipSignature -Path $archive)) {
    throw "Downloaded SoX archive is not a ZIP file: $archive"
}

Write-Host "Extracting SoX into $toolRoot"
Expand-Archive -Path $archive -DestinationPath $toolRoot -Force

$repoSox = Get-ChildItem -Path $toolRoot -Recurse -Filter sox.exe -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $repoSox) {
    throw "SoX download completed, but sox.exe was not found under $toolRoot."
}

Write-Host "Installed repo-local SoX: $($repoSox.FullName)"
