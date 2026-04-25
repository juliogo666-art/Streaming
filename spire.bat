@echo off
title Lanzador Galeno
setlocal

:: 1. Entramos en la carpeta del proyecto
cd /d "%~dp0"

echo Verificando dependencias...
:: 2. Sincroniza el entorno (Instala lo que falte segun el pyproject.toml)
call uv sync

echo Iniciando aplicacion...
:: 3. Ejecutamos la app
call uv run main.py  