@echo off
setlocal enabledelayedexpansion

set "input_dir=%cd%"
if not "%1" == "" (
    set "input_dir=%~f1"
)

for %%f in ("%input_dir%\*.dds") do (
    set "output_file=!cd!\%%~nf.png"
    magick "%%f" "!output_file!"
)
