param(
    [string]$Python = "python",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8086,
    [int]$TimeoutSeconds = 900
)

$ErrorActionPreference = "Stop"
$env:AI_COMPANION_ENABLE_LOCAL_MODELS = "1"
$env:AI_COMPANION_PRELOAD_LOCAL_MODELS = "1"
$env:AI_COMPANION_STRICT_LOCAL_MODELS = "1"
$env:AI_COMPANION_SMOKE_TEST_LOCAL_MODELS = "1"
$env:USE_TF = "0"

. (Join-Path $PSScriptRoot "use_repo_sox.ps1")

$args = @("-m", "uvicorn", "server.main:app", "--host", $HostName, "--port", "$Port")
$logDir = "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "local-model-server.out.log"
$stderr = Join-Path $logDir "local-model-server.err.log"
$process = Start-Process `
    -WindowStyle Hidden `
    -PassThru `
    -FilePath $Python `
    -ArgumentList $args `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr
Write-Host "Started local model server process $($process.Id). Waiting for model preload and health..."

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do {
    Start-Sleep -Seconds 5
    if ($process.HasExited) {
        if (Test-Path $stdout) {
            Write-Host "--- stdout ---"
            Get-Content $stdout
        }
        if (Test-Path $stderr) {
            Write-Host "--- stderr ---"
            Get-Content $stderr
        }
        throw "Server process exited before becoming healthy. Exit code: $($process.ExitCode)"
    }
    try {
        Invoke-RestMethod -Uri "http://${HostName}:${Port}/api/health" -TimeoutSec 5 | Out-Null
        Write-Host "Server is healthy. Current GPU status:"
        nvidia-smi
        exit 0
    } catch {
        Write-Host "Still waiting for local models to preload..."
    }
} while ((Get-Date) -lt $deadline)

throw "Timed out waiting for local model server health after $TimeoutSeconds seconds."
