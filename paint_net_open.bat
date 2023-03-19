@echo off

if "%~1"=="" (
    echo Usage: %0 folder1 folder2
    exit /b 1
)

setlocal enabledelayedexpansion

set "folder1=%~1"
set "folder2=%~2"

for /f "delims=" %%f in ('dir /b /a-d "%folder1%\*"') do (
    set "name=%%~nf"
    set "ext=%%~xf"
    set "file1=%folder1%\%%f"
    set "file2=%folder2%\!name!.*"
    for %%a in ("!file2!") do (set "file2=%%a")
    if exist "!file2!" (
        echo Opening !file1! and !file2! in Paint.NET...
        start "" paintdotnet:"!file1!"
        start "" paintdotnet:"!file2!"
        pause
    )
)

endlocal
