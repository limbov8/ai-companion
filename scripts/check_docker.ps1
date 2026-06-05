$ErrorActionPreference = "Stop"

try {
    docker info | Out-Null
} catch {
    Write-Host ""
    Write-Host "Docker is not running or the Docker Desktop Linux engine is unavailable."
    Write-Host "Start Docker Desktop, wait until it says it is running, then retry this command."
    Write-Host ""
    Write-Host "For local model/API work without Postgres, use:"
    Write-Host "  make run-no-db"
    Write-Host "  make run-check-no-db"
    Write-Host ""
    exit 1
}
