param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "Installing Qwen ASR runtime dependencies..."
& $Python -m pip install `
    "nagisa==0.2.11" `
    "soynlp==0.0.493" `
    "accelerate==1.12.0" `
    "qwen-omni-utils" `
    "librosa" `
    "soundfile" `
    "sox" `
    "gradio" `
    "flask" `
    "pytz"

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Installing qwen-asr without its transformers pin so qwen-tts can keep transformers 4.57.3..."
& $Python -m pip install "qwen-asr==0.0.6" --no-deps

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
