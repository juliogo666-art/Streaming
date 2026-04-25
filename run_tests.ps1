# ============================================================================
# Ejecutar tests unitarios y generar informe de cobertura HTML
# ============================================================================
# Uso:
#   .\run_tests.ps1           → Ejecuta tests + abre reporte HTML
#   .\run_tests.ps1 -NoBrowser → Ejecuta tests sin abrir el navegador
# ============================================================================

param(
    [switch]$NoBrowser
)

Write-Host ""
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  SPIRE Streaming — Suite de Tests Unitarios" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Ejecuta pytest con coverage (terminal + HTML)
uv run pytest --cov --cov-report=html --cov-report=term

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Todos los tests pasaron correctamente." -ForegroundColor Green
    Write-Host "Reporte HTML generado en: ./htmlcov/index.html" -ForegroundColor Green

    if (-not $NoBrowser) {
        Start-Process "./htmlcov/index.html"
    }
} else {
    Write-Host ""
    Write-Host "Algunos tests han fallado. Revisa la salida anterior." -ForegroundColor Yellow
}
