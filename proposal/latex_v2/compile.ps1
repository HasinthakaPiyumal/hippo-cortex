param([switch]$Rasterise = $true)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$buildDir = Join-Path $here 'build'
if (!(Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }

# --- Pick LaTeX engine ---
$engine = $null
if (Get-Command tectonic -ErrorAction SilentlyContinue) { $engine = 'tectonic' }
elseif (Get-Command latexmk  -ErrorAction SilentlyContinue) { $engine = 'latexmk' }
else { Write-Error 'No LaTeX engine found (tectonic or latexmk required).'; exit 1 }

Write-Host "[compile] using $engine" -ForegroundColor Cyan

if ($engine -eq 'tectonic') {
    # Tectonic auto-fetches packages; --keep-intermediates lets us re-use .bcf for debugging
    & tectonic -X compile main.tex --outdir $buildDir --keep-logs --keep-intermediates
    if ($LASTEXITCODE -ne 0) { Write-Error "Tectonic failed."; exit $LASTEXITCODE }
} else {
    & latexmk -pdf -interaction=nonstopmode -file-line-error -outdir=$buildDir main.tex
    if ($LASTEXITCODE -ne 0) { Write-Error "latexmk failed."; exit $LASTEXITCODE }
}

$pdfPath = Join-Path $buildDir 'main.pdf'
if (!(Test-Path $pdfPath)) { Write-Error "Expected PDF not found: $pdfPath"; exit 1 }

# --- Page count ---
$pdfinfo  = 'C:\poppler\poppler-24.08.0\Library\bin\pdfinfo.exe'
$pdftoppm = 'C:\poppler\poppler-24.08.0\Library\bin\pdftoppm.exe'
if (Test-Path $pdfinfo) {
    $info = & $pdfinfo $pdfPath
    Write-Host "[compile] $info" -ForegroundColor Yellow
}

# --- Rasterise pages for visual validation ---
if ($Rasterise -and (Test-Path $pdftoppm)) {
    Get-ChildItem $buildDir -Filter 'page-*.png' -ErrorAction SilentlyContinue | Remove-Item -Force
    & $pdftoppm -r 150 -png $pdfPath (Join-Path $buildDir 'page')
    Write-Host "[compile] rasterised pages -> $buildDir\page-*.png" -ForegroundColor Cyan
}

# --- Copy final PDF to proposal/ ---
$finalName = 'Group07_Milestone_1_2026-04-26.pdf'
$finalPath = Join-Path (Split-Path -Parent $here) $finalName
Copy-Item $pdfPath $finalPath -Force
Write-Host "[compile] published -> $finalPath" -ForegroundColor Green
