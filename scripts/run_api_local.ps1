param(
    [string]$Python = "python",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"
$env:AI_COMPANION_ENABLE_LOCAL_MODELS = "1"
$env:AI_COMPANION_PRELOAD_LOCAL_MODELS = "1"
$env:AI_COMPANION_STRICT_LOCAL_MODELS = "1"
$env:AI_COMPANION_SMOKE_TEST_LOCAL_MODELS = "1"
$env:USE_TF = "0"

. (Join-Path $PSScriptRoot "use_repo_sox.ps1")

& $Python -m uvicorn server.main:app --host $HostName --port $Port
