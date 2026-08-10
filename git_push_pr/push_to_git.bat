@echo off
REM ============================================
REM  Git Push Script - SVHN House Number Recognition
REM  Group 57 - Computer Vision - USTH
REM ============================================
REM
REM  HUONG DAN SU DUNG:
REM  1. Tao repository tren GitHub/GitLab
REM  2. Sua REMOTE_URL o duoi thanh URL repo cua ban
REM  3. Chay script nay: double-click hoac chay trong cmd
REM ============================================

echo.
echo ==========================================
echo   SVHN House Number Recognition - Git Push
echo   Group 57 - Computer Vision - USTH
echo ==========================================
echo.

REM ===== CAU HINH =====
REM Thay URL nay bang repo cua ban
set REMOTE_URL=https://github.com/DTKien2005/Street-View-House-Numbers-Recognition.git
set BRANCH=main

REM ===== Di chuyen vao thu muc project =====
cd /d "%~dp0\.."

echo [1/6] Copying .gitignore vao project root...
copy /Y "%~dp0\.gitignore" ".gitignore" >nul 2>&1

echo [2/6] Initializing git repository...
git init

echo [3/6] Setting up remote...
git remote remove origin >nul 2>&1
git remote add origin %REMOTE_URL%

echo [4/6] Adding files...
git add .

echo [5/6] Creating commit...
git commit -m "feat: Street View House Numbers Recognition - YOLO26 + HOG/SVM pipeline"

echo [6/6] Pushing to remote...
git branch -M %BRANCH%
git push -u origin %BRANCH%

echo.
echo ==========================================
echo   DONE! Check your repository at:
echo   %REMOTE_URL%
echo ==========================================
echo.

pause
