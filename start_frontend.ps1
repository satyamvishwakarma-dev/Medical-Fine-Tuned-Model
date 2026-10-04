# =====================================================================
# BioMistral Medical Q&A - Frontend Launcher
# Launches the medical web app at http://localhost:8000
# =====================================================================

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " 🩺 Launching BioMistral Medical Consultation Web App" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

# Check if Ollama is running
try {
    $ollamaCheck = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction Stop
    $hasModel = $ollamaCheck.models | Where-Object { $_.name -like "*biomistral-med*" }
    if ($hasModel) {
        Write-Host " [✓] Ollama connected & 'biomistral-med' model verified." -ForegroundColor Green
    } else {
        Write-Host " [!] Ollama running, but 'biomistral-med' was not found in 'ollama list'." -ForegroundColor Yellow
        Write-Host "     Run: ollama create biomistral-med -f .\src\medical_fine_tuned_model\model\Modelfile" -ForegroundColor Yellow
    }
} catch {
    Write-Host " [!] Warning: Ollama does not seem to be responding on port 11434." -ForegroundColor Yellow
    Write-Host "     Ensure the Ollama desktop app is started." -ForegroundColor Yellow
}

Write-Host "`n Starting FastAPI Web Server at http://localhost:8000..." -ForegroundColor Cyan
Write-Host " Press Ctrl+C in this console to stop the server.`n" -ForegroundColor Gray

# Open browser automatically after launching
Start-Process "http://localhost:8000"

# Run server with project virtual environment
.\.venv\Scripts\python.exe server.py
