@echo off
echo ===============================================
echo    Starting Stress Predictor UI Application
echo ===============================================
echo.

REM Step 1: Go to project folder
cd /d C:\Users\ayush\OneDrive\Desktop\Stress_Predictor_UI

echo Checking Python installation...
py --version
if %errorlevel% neq 0 (
    echo Python not found! Install Python before running this script.
    pause
    exit
)

echo.
echo Installing required packages...
py -m pip install -r requirements.txt

echo.
echo Running Streamlit app...
py -m streamlit run app.py

echo.
echo ===============================================
echo        Application Closed
echo ===============================================
pause
