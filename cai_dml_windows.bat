@echo off
rem ===================================================================
rem  Cai moi truong chay THU bang card lien (Intel HD 620 / UHD / AMD)
rem
rem  Chay: bam doi vao file nay, hoac go  cai_dml_windows.bat  o cmd.
rem
rem  No KHONG dung toi ban Python dang co cua ban. No tao mot thu muc
rem  rieng .venv-dml va cai torch-directml vao do thoi.
rem ===================================================================
setlocal
cd /d "%~dp0"
echo.
echo === Cai moi truong chay bang card lien (DirectML) ===
echo.

if defined VIRTUAL_ENV (
  echo CANH BAO: dang o trong mot moi truong ao roi.
  echo           Go  deactivate  roi chay lai file nay cho sach.
  echo.
)

rem --- 1. Tim mot ban Python tu 3.12 tro xuong -----------------------
set PYCMD=
call :thu 3.12
call :thu 3.11
call :thu 3.10
call :thu 3.9
if "%PYCMD%"=="" goto thieu_python
echo [1/4] Dung %PYCMD%

rem --- 2. Tao moi truong ao -----------------------------------------
if exist .venv-dml (
  echo [2/4] Xoa .venv-dml cu
  rmdir /s /q .venv-dml
)
echo [2/4] Tao .venv-dml
%PYCMD% -m venv .venv-dml
if errorlevel 1 goto loi
call ".venv-dml\Scripts\activate.bat"

rem --- 3. Cai goi ---------------------------------------------------
echo [3/4] Cai torch-directml (hon 500 MB, doi mot lat)
python -m pip install --upgrade pip --quiet
python -m pip install torch-directml numpy
if errorlevel 1 goto loi_goi

rem --- 4. Thu may ---------------------------------------------------
echo.
echo [4/4] Thu xem may chay duoc den dau
echo.
python -m turbo.tools.check --device dml
echo.
echo ===================================================================
echo  Lan sau muon chay lai, mo cmd o thu muc nay roi go:
echo.
echo      .venv-dml\Scripts\activate
echo      python -m turbo.tools.check --device dml
echo      python -m turbo.tools.measure speed --device dml
echo ===================================================================
goto xong

:thu
if not "%PYCMD%"=="" goto :eof
py -%1 -c "import sys" >nul 2>&1
if errorlevel 1 goto :eof
set PYCMD=py -%1
goto :eof

:thieu_python
echo.
echo KHONG TIM THAY ban Python nao tu 3.12 tro xuong.
echo.
echo torch-directml chi co goi cho Python 3.8 den 3.12, va no ghim
echo torch==2.4.1 - ban torch do cung chi co toi 3.12. Ban dang co:
echo.
py -0
echo.
echo Cai them Python 3.12 - KHONG lam hong ban dang co, hai ban chay
echo song song duoc. Chon mot trong hai cach:
echo.
echo   Cach 1 (nhanh):   winget install Python.Python.3.12
echo     Bao khong nhan ten do thi go:  winget search Python.Python
echo.
echo   Cach 2: vao python.org/downloads, tai "Python 3.12.x" ban
echo     "Windows installer (64-bit)", cai binh thuong.
echo.
echo Cai xong thi MO CMD MOI roi chay lai file nay.
goto xong

:loi_goi
echo.
echo Cai goi that bai. Thay "from versions: none" thi ban Python nay van
echo qua moi - xem turbo\docs\INTEL_620.md.
goto xong

:loi
echo Khong tao duoc moi truong ao.
goto xong

:xong
echo.
pause
