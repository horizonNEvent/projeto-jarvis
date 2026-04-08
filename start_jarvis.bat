@echo off
title Jarvis - Assistente de Voz
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m jarvis.main
pause
