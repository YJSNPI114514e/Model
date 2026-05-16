# GRIM GPU 環境: CUDA 版 PyTorch を .venv に入れる
# 要: NVIDIA GPU + ドライバ
Set-Location $PSScriptRoot

Write-Host "=== GRIM GPU setup (CUDA 12.4) ===" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$py = ".\.venv\Scripts\python.exe"
$ErrorActionPreference = "Continue"

& $py -m pip install --upgrade pip 2>&1 | Out-Host
& $py -m pip uninstall -y torch torchvision torchaudio 2>&1 | Out-Host

Write-Host "Installing CUDA PyTorch (cu124) ..."
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $py -m pip install torchdiffeq tqdm numpy datasets huggingface_hub 2>&1 | Out-Host

$ErrorActionPreference = "Stop"

Write-Host "Checking GPU ..."
& $py -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

Write-Host ""
Write-Host "Train example:" -ForegroundColor Green
Write-Host "  python scripts\train.py --dataset fn-aka-mur/wiki40b_ja --max-samples 50000 --fast --device cuda --amp"
