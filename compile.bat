@echo off
echo Compiling thesis LaTeX document using Tectonic...
.\.venv\Scripts\tectonic.exe -o docs\thesis docs\thesis\thesis_template\thesis.tex
if %errorlevel% neq 0 (
    echo Compilation failed!
    exit /b %errorlevel%
)
echo Compilation successful! PDF generated at docs\thesis\thesis.pdf
