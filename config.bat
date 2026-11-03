@echo off
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
echo.
echo Installation terminee. Pour lancer le projet : python main.py
cmd /k
