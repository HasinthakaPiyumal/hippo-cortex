@echo off
rem Run from anywhere: paths resolve from this script's folder (docs\thesis).
set "THESIS_DIR=%~dp0"
set "REPO_ROOT=%THESIS_DIR%..\.."
echo Compiling thesis LaTeX document using Tectonic...
"%REPO_ROOT%\.venv\Scripts\tectonic.exe" -o "%THESIS_DIR%." "%THESIS_DIR%thesis_template\thesis.tex"
if %errorlevel% neq 0 (
    echo Compilation failed!
    exit /b %errorlevel%
)
echo Compilation successful! PDF generated at %THESIS_DIR%thesis.pdf
