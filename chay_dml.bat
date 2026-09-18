@echo off
rem ===================================================================
rem  Chay thu bang card lien. Cai xong roi thi dung file nay.
rem  (Phai chay cai_dml_windows.bat mot lan truoc do.)
rem
rem    chay_dml.bat              thu xem may chay duoc den dau
rem    chay_dml.bat do           do toc do card lien roi do lai tren CPU
rem    chay_dml.bat --only 4     chi chay muc 4
rem    chay_dml.bat --fan cpu    thu cach gom tia khac
rem  (moi thu khac deu duoc chuyen thang cho bo kiem tra)
rem ===================================================================
setlocal
cd /d "%~dp0"

set "VENV=%LOCALAPPDATA%\robotmini-dml"
if exist ".venv-dml\Scripts\activate.bat" set "VENV=%CD%\.venv-dml"

if not exist "%VENV%\Scripts\activate.bat" (
  echo Chua thay moi truong ao o:
  echo   %VENV%
  echo.
  echo Chay cai_dml_windows.bat truoc da.
  echo.
  pause
  exit /b 1
)
call "%VENV%\Scripts\activate.bat"

if /i "%~1"=="do" goto doc

rem %* la MOI tham so ban go them, chuyen het cho bo kiem tra.
python -m turbo.tools.check --device dml %*
echo.
echo Muon do toc do thi go:  chay_dml.bat do
echo.
pause
exit /b 0

:doc
echo === Do toc do tren CARD LIEN ===
python -m turbo.tools.measure speed --device dml --pop 64 --no-v1
echo.
echo === Do lai tren CPU de so ===
python -m turbo.tools.measure speed --device cpu --pop 64
echo.
pause
exit /b 0
