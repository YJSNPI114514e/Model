"""
グローバル Python の壊れた PyTorch を避け、.venv を優先する。
nvfuser_codegen.dll エラーは CUDA 版 PyTorch の不整合で起きることが多い。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"


def reexec_in_venv_if_needed() -> None:
    """プロジェクト .venv があれば、そちらの Python で再実行する。"""
    if os.environ.get("GRIM_NO_VENV_REEXEC"):
        return
    if not VENV_PY.is_file():
        return
    try:
        if Path(sys.executable).resolve() == VENV_PY.resolve():
            return
    except OSError:
        return
    os.execv(str(VENV_PY), [str(VENV_PY), *sys.argv])


def ensure_torch() -> None:
    """import torch を試し、失敗時は修復手順を表示して終了。"""
    try:
        import torch  # noqa: F401
    except OSError as exc:
        msg = str(exc)
        if "nvfuser" in msg.lower() or "WinError 127" in msg:
            print(
                "PyTorch の DLL 読み込みに失敗しました（CUDA 版の不整合が原因のことが多いです）。\n"
                "次のいずれかを実行してください:\n"
                f"  1) PowerShell: cd {ROOT}\n"
                "     .\\setup_env.ps1\n"
                "  2) 手動:\n"
                "     python -m venv .venv\n"
                "     .\\.venv\\Scripts\\Activate.ps1\n"
                "     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
                "     pip install -r requirements.txt\n"
                "  3) 以降は .venv の Python を使う:\n"
                "     .\\.venv\\Scripts\\python.exe run_demo.py\n"
            )
        else:
            print(f"PyTorch の import に失敗しました: {exc}")
        sys.exit(1)
