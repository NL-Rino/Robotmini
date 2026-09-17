@echo off
rem ===================================================================
rem  Cai moi truong chay THU bang card lien (Intel HD 620 / UHD / AMD)
rem
rem  Chay: bam doi vao file nay, hoac go  cai_dml_windows.bat  o cmd.
rem
rem  No KHONG dung toi ban Python dang co cua ban.
rem
rem  Moi truong ao dat o  %LOCALAPPDATA%\robotmini-dml  chu KHONG dat canh
rem  du an. Ly do: Windows chi cho duong dan dai 260 ky tu, ma goi torch co
rem  cay thu muc sau toi ~150 ky tu. Du an nam trong OneDrive thi rieng
rem  phan dau da hon 90 ky tu, cong vao la tran -> pip bao
rem  "[Errno 22] Invalid argument" giua chung.
rem ===================================================================
setlocal
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\robotmini-dml"

echo.
echo === Cai moi truong chay bang card lien (DirectML) ===
echo.
echo  Du an     : %CD%
echo  Moi truong: %VENV%
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
if exist ".venv-dml" (
  echo       Xoa .venv-dml cu trong thu muc du an ^(dat sai cho^)
  rmdir /s /q ".venv-dml"
)
if exist "%VENV%" (
  echo       Xoa moi truong cu
  rmdir /s /q "%VENV%"
)
echo [2/4] Tao moi truong ao
%PYCMD% -m venv "%VENV%"
if errorlevel 1 goto loi
call "%VENV%\Scripts\activate.bat"

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
echo  Xong. Lan sau chi can bam doi vao  chay_dml.bat
echo  ^(hoac  chay_dml.bat do  de do toc do^)
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
echo CAI GOI THAT BAI.
echo.
echo   - Thay "[Errno 22] Invalid argument" hay "path too long":
echo     duong dan van qua dai. Thu chep ca thu muc du an ra mot cho
echo     ngan hon, vi du  C:\robot\  roi chay lai file nay.
echo.
echo   - Thay "from versions: none": ban Python van qua moi.
echo.
echo   - Thay loi ve quyen hay file dang mo: OneDrive co the dang khoa
echo     file. Bam chuot phai vao bieu tuong OneDrive ^> Pause syncing,
echo     roi chay lai.
echo.
echo   Chi tiet: turbo\docs\INTEL_620.md
goto xong

:loi
echo Khong tao duoc moi truong ao o %VENV%
goto xong

:xong
echo.
pause
