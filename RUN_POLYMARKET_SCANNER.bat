@echo off
echo ========================================
echo  POLYMARKET SWARM TRADER v3.0
echo  6-Model LLM Consensus Scanner
echo ========================================
echo.
echo Choose scan type:
echo   1. OPTIMIZED (200 markets, 2-phase) - RECOMMENDED
echo   2. Standard (30 markets, full analysis)
echo.
set /p choice="Enter choice [1/2]: "

cd /d C:\Users\tomma\PolyAgent\scripts\polymarket_swarm

if "%choice%"=="2" (
    echo.
    echo Running Standard Scanner...
    python run_v3.py
) else (
    echo.
    echo Running Optimized Scanner...
    python run_optimized.py
)

echo.
echo ========================================
echo  Scan complete!
echo ========================================
pause
