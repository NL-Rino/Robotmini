@echo off
rem ===================================================================
rem  Chay thu bang card lien. Cai xong roi thi dung file nay.
rem  (Phai chay cai_dml_windows.bat mot lan truoc do.)
rem
rem    chay_dml.bat              thu xem may chay duoc den dau
rem    chay_dml.bat do           do toc do card lien roi do lai tren CPU
rem    chay_dml.bat cahai        do card + CPU CUNG LAM so voi tung cai mot
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
if /i "%~1"=="cahai" goto cahai

rem %* la MOI tham so ban go them, chuyen het cho bo kiem tra.
python -m turbo.tools.check --device dml %*
echo.
echo Muon do toc do thi go:  chay_dml.bat do
echo.
pause
exit /b 0

:cahai
echo === Card lien + CPU cung lam, so voi tung cai mot ===
python -m turbo.tools.measure duo --device dml,cpu --pop 64 --maps 1
echo.
echo Neu dong cuoi bao "khong hon" thi thu chua bot luong CPU lai:
echo     python -m turbo.tools.measure duo --device dml,cpu --pop 64 --maps 1 --threads 1
echo.
pause
exit /b 0

:doc
echo === CARD LIEN: quan the cang to thi duoc bao nhieu ===
python -m turbo.tools.measure scale --device dml --pops 16,64,256 --maps 1
echo.
echo === CPU: cung bang do de so ===
python -m turbo.tools.measure scale --device cpu --pops 16,64,256 --maps 1
echo.
echo === Mot THE HE that: card lien so voi ban "tung xe mot" ===
python -m turbo.tools.measure speed --device dml --pop 64 --maps 1 --no-v1
python -m turbo.tools.measure speed --device cpu --pop 64 --maps 1
echo.
pause
exit /b 0
