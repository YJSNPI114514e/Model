# GRIM setup: install CPU PyTorch into .venv (avoids nvfuser DLL errors)
Set-Location $PSScriptRoot

Write-Host "=== GRIM setup ===" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv ..."
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "ERROR: .venv python not found." -ForegroundColor Red
    exit 1
}

# pip warnings go to stderr; Stop would treat them as fatal errors
$ErrorActionPreference = "Continue"

Write-Host "Upgrading pip ..."
& $py -m pip install --upgrade pip 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$torchCheck = & $py -m pip show torch 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Removing old PyTorch in venv ..."
    & $py -m pip uninstall -y torch torchvision torchaudio 2>&1 | Out-Host
}

Write-Host "Installing CPU PyTorch ..."
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing torchdiffeq, tqdm, numpy, datasets, gradio ..."
& $py -m pip install torchdiffeq tqdm numpy datasets huggingface_hub gradio 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ErrorActionPreference = "Stop"

Write-Host "Smoke test ..."
& $py run_demo.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Done. Use:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python run_demo.py"
Write-Host "  python scripts\train.py --data data\sample_corpus.txt"
Write-Host "Or: run.bat run_demo.py"
