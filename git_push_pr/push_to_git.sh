#!/bin/bash
# ============================================
#  Git Push Script - SVHN House Number Recognition
#  Group 57 - Computer Vision - USTH
# ============================================
#
#  HUONG DAN SU DUNG:
#  1. Tao repository tren GitHub/GitLab
#  2. Sua REMOTE_URL o duoi thanh URL repo cua ban
#  3. chmod +x push_to_git.sh && ./push_to_git.sh
# ============================================

set -e

echo ""
echo "=========================================="
echo "  SVHN House Number Recognition - Git Push"
echo "  Group 57 - Computer Vision - USTH"
echo "=========================================="
echo ""

# ===== CAU HINH =====
# Thay URL nay bang repo cua ban
REMOTE_URL="https://github.com/YOUR_USERNAME/Street-View-House-Numbers-Recognition.git"
BRANCH="main"

# ===== Di chuyen vao thu muc project =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "[1/6] Copying .gitignore vao project root..."
cp -f "$SCRIPT_DIR/.gitignore" ".gitignore"

echo "[2/6] Initializing git repository..."
git init

echo "[3/6] Setting up remote..."
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"

echo "[4/6] Adding files..."
git add .

echo "[5/6] Creating commit..."
git commit -m "feat: Street View House Numbers Recognition - YOLO26 + HOG/SVM pipeline"

echo "[6/6] Pushing to remote..."
git branch -M "$BRANCH"
git push -u origin "$BRANCH"

echo ""
echo "=========================================="
echo "  DONE! Check your repository at:"
echo "  $REMOTE_URL"
echo "=========================================="
echo ""
