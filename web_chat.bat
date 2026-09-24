@echo off
title Micro-AGI Web Studio Server
cls
cd /d "%~dp0"
echo Starting Micro-AGI Web Studio...
echo Open your browser at: http://127.0.0.1:8765
start http://127.0.0.1:8765
python web_chat_server.py
pause
