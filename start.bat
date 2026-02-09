@echo off
set CONDA_ENV=poetry-ocr

REM === Инициализация conda ===
call "%USERPROFILE%\anaconda3\Scripts\activate.bat"

REM === Запуск backend ===
call conda activate %CONDA_ENV%
start cmd /k "cd core && python app.py"

REM === Запуск frontend ===
start cmd /k "cd service && npm run dev"