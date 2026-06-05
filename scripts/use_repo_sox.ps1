param(
    [string]$Root = ""
)

if (-not $Root) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$repoSox = Get-ChildItem -Path (Join-Path $Root ".tools\sox") -Recurse -Filter sox.exe -ErrorAction SilentlyContinue |
    Select-Object -First 1

if ($repoSox) {
    $soxDir = $repoSox.Directory.FullName
    if ($env:PATH -notlike "*$soxDir*") {
        $env:PATH = "$soxDir;$env:PATH"
    }
    Write-Host "Using repo-local SoX: $($repoSox.FullName)"
} elseif (-not (Get-Command sox -ErrorAction SilentlyContinue)) {
    Write-Warning "SoX is not available. Run 'make install-audio-tools' before strict local TTS startup."
}
