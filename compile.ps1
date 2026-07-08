Write-Host "Compiling thesis LaTeX document using Tectonic..." -ForegroundColor Cyan
& ".\.venv\Scripts\tectonic.exe" -o docs\thesis docs\thesis\thesis_template\thesis.tex
if ($LASTEXITCODE -ne 0) {
    Write-Error "Compilation failed!"
    exit $LASTEXITCODE
}
Write-Host "Compilation successful! PDF generated at docs\thesis\thesis.pdf" -ForegroundColor Green
