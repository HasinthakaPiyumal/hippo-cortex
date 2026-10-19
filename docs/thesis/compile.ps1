# Run from anywhere: paths resolve from this script's folder (docs\thesis).
$ThesisDir = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ThesisDir "..\..")
Write-Host "Compiling thesis LaTeX document using Tectonic..." -ForegroundColor Cyan
& (Join-Path $RepoRoot ".venv\Scripts\tectonic.exe") -o $ThesisDir (Join-Path $ThesisDir "thesis_template\thesis.tex")
if ($LASTEXITCODE -ne 0) {
    Write-Error "Compilation failed!"
    exit $LASTEXITCODE
}
Write-Host "Compilation successful! PDF generated at $ThesisDir\thesis.pdf" -ForegroundColor Green
