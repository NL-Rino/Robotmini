@echo off
rem ===================================================================
rem  Tai ban moi nhat ve, de len ban dang co.
rem
rem  Co git thi no dung git. Khong co thi no tai file nen tu GitHub roi
rem  chep de len - chi chep ma nguon, khong dung toi runs\ hay brains\.
rem
rem  Cach chac chan nhat van la cai git mot lan cho xong:
rem      winget install Git.Git
rem ===================================================================
setlocal
cd /d "%~dp0"
set "REPO=NL-Rino/Robotmini"
set "BRANCH=claude/inspiring-noether-9n5p0d"

echo.
echo === Cap nhat ma nguon ===
echo.

where git >nul 2>&1
if errorlevel 1 goto taizip
if not exist ".git" goto taizip
echo Dung git
git pull
goto xong

:taizip
echo Khong co git (hoac thu muc nay khong phai ban sao git).
echo Tai file nen tu GitHub...
set "ZIP=%TEMP%\robotmini_update.zip"
set "OUT=%TEMP%\robotmini_update"
if exist "%OUT%" rmdir /s /q "%OUT%"
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; try { Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/%REPO%/archive/refs/heads/%BRANCH%.zip' -OutFile '%ZIP%' } catch { exit 1 }"
if errorlevel 1 goto khongtai
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; Expand-Archive -Path '%ZIP%' -DestinationPath '%OUT%' -Force"
if errorlevel 1 goto khongtai

set "SRC="
for /d %%D in ("%OUT%\*") do set "SRC=%%D"
if "%SRC%"=="" goto khongtai

echo Chep de len ban hien tai...
robocopy "%SRC%" "%CD%" /E /NFL /NDL /NJH /NJS /NP /XF capnhat.bat >nul
if errorlevel 8 goto khongchep
echo Xong.
echo (File capnhat.bat khong tu de len chinh no duoc - neu no doi thi
echo  tai tay mot lan.)
goto xong

:khongtai
echo.
echo KHONG TAI DUOC.
echo   - Kho ma nguon co the dang de rieng tu: mo trinh duyet, dang nhap
echo     GitHub, vao trang nhanh roi bam Code ^> Download ZIP.
echo   - Hoac cai git cho xong:  winget install Git.Git
echo     Cai xong mo CMD MOI roi chay lai file nay.
goto xong

:khongchep
echo Chep that bai. Dong het chuong trinh dang mo file trong thu muc nay
echo roi thu lai.

:xong
echo.
pause
