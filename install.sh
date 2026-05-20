#!/usr/bin/env bash
# DartsCamera – Installation (Linux / macOS / Raspberry Pi)
set -e

echo "=== DartsCamera – Installation ==="

# Python check
if ! command -v python3 &>/dev/null; then
  echo "Python 3 est requis. Installez-le avec votre gestionnaire de paquets."
  exit 1
fi

# Venv
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "Environnement virtuel créé (.venv)"
fi

source .venv/bin/activate

# Dépendances système pour OpenCV sur Raspberry Pi / Debian
if command -v apt-get &>/dev/null; then
  echo "Installation des dépendances système OpenCV..."
  sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 2>/dev/null || true
fi

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Installation terminée ==="
echo ""
echo "Lancement :"
echo "  source .venv/bin/activate"
echo "  python app.py"
echo ""
echo "Ouvrez http://localhost:5000 dans votre navigateur."
