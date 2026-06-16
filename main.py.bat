@echo off
setlocal EnableDelayedExpansion
:: ==============================================================================
::  STRICT OFFLINE MODE ENFORCEMENT
:: ==============================================================================
:: 1. Dynamically set paths relative to this .bat file's location (Fixes hardcoded username)
set "PROJECT_DIR=%~dp0"
set "HF_HOME=%PROJECT_DIR%model_cache"
set "XDG_CACHE_HOME=%HF_HOME%"
set "MODEL_CACHE_DIR=%HF_HOME%"

:: 2. Block ALL known Python library and CLI internet callbacks
set TRANSFORMERS_OFFLINE=1
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
set MARKER_OFFLINE=1
set OLLAMA_NO_UPDATE=1

:: 3. Network Nullification
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
set NO_PROXY=*

:: 4. Python Optimization
set PYTHONUNBUFFERED=1

:: ==============================================================================
::  EXECUTION SETUP
:: ==============================================================================
echo ============================================================
echo  ???  STRICT OFFLINE MODE ACTIVATED
echo  ??  Model Cache Directory: %HF_HOME%
echo ============================================================
echo.

cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    echo ? Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ? ERROR: Virtual environment 'venv' not found!
    pause
    exit /b 1
)

:: ==============================================================================
::  PIPELINE EXECUTION
:: ==============================================================================
echo.
echo [STEP 1] Converting PDFs to Markdown (Marker CLI)...
python FirstStep_Convertor.py -i .\input -o .\output_markdown
if errorlevel 1 goto :error_step1

echo.
echo [STEP 2] Chunking Markdown files...
python SecondStep_Chunker.py -i .\output_markdown -o .\chunked_output
if errorlevel 1 goto :error_step2

echo.
echo [STEP 3] Embedding and Storing in Qdrant...
python ThirdStep_embed_and_store.py .\chunked_output -d .\vector_db
if errorlevel 1 goto :error_step3

echo.
echo [STEP 4] Running RAG Query Pipeline...
python FourthStep_query.py -d .\vector_db
if errorlevel 1 goto :error_step4

echo.
echo ============================================================
echo  ??  PIPELINE COMPLETED SUCCESSFULLY IN STRICT OFFLINE MODE!
echo ============================================================
pause
exit /b 0

:: ==============================================================================
::  ERROR HANDLING
:: ==============================================================================
:error_step1
echo.
echo ? STEP 1 FAILED.
pause
exit /b 1

:error_step2
echo.
echo ? STEP 2 FAILED.
echo Possible Offline Cause: Tokenizer not found in '%HF_HOME%'.
echo Ensure 'intfloat/multilingual-e5-large' is downloaded to the model_cache folder.
pause
exit /b 1

:error_step3
echo.
echo ? STEP 3 FAILED.
pause
exit /b 1

:error_step4
echo.
echo ? STEP 4 FAILED.
pause
exit /b 1