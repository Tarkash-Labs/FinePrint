Write-Host "Starting FinePrint Servers..." -ForegroundColor Green

# Start Backend in a new window
Write-Host "Starting FastAPI Backend..."
Start-Process powershell -ArgumentList "-NoExit -Command `"cd backend; if (Test-Path '../.venv/Scripts/activate.ps1') { . '../.venv/Scripts/activate.ps1' }; python -m uvicorn main:app --reload`""

# Start Frontend in a new window
Write-Host "Starting React Frontend..."
Start-Process powershell -ArgumentList "-NoExit -Command `"cd frontend; npm run dev`""

Write-Host "Both servers are starting in separate windows!" -ForegroundColor Blue
