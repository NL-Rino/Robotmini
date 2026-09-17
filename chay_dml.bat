@echo off
rem ===================================================================
rem  Chay thu bang card lien. Cai xong roi thi dung file nay.
rem  (Phai chay cai_dml_windows.bat mot lan truoc do.)
rem ===================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv-dml\Scripts\activate.bat" (
  echo Chua co .venv-dml. Chay cai_dml_windows.bat truoc da.
  echo.
  pause
  exit /b 1
)
call ".venv-dml\Scripts\activate.bat"

if "%~1"=="" goto kiemtra
if /i "%~1"=="do" goto doc
goto kiemtra

:kiemtra
python -m turbo.tools.check --device dml
echo.
echo Muon do toc do thi go:  chay_dml.bat do
echo.
pause
exit /b 0

:doc
echo === Do toc do tren card lien ===
python -m turbo.tools.measure speed --device dml --pop 64 --no-v1
echo.
echo === Do lai tren CPU de so ===
python -m turbo.tools.measure speed --device cpu --pop 64
echo.
pause
exit /b 0
