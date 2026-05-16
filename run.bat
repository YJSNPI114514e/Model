@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

if "%~1" == "" (
  echo Starting WebUI...
  %PYTHON% scripts\webui.py
  goto :end
)

if /i "%~1" == "generate" (
  shift
  %PYTHON% scripts\generate.py %*
  goto :end
)

if /i "%~1" == "train" (
  shift
  %PYTHON% scripts\train.py %*
  goto :end
)

if /i "%~1" == "webui" (
  shift
  %PYTHON% scripts\webui.py %*
  goto :end
)

%PYTHON% %*

:end
if errorlevel 1 (
  echo.
  echo Execution failed with error level %errorlevel%.
  pause
)
endlocal
