@echo off
setlocal enabledelayedexpansion enableextensions

REM Define an array of batch script names
set scripts=extract.bat clean.bat update_files.py

REM Define the absolute path where the shortcuts will be created
set "shortcut_path=%cd%"

REM Loop through the array and create a shortcut for each script
for %%a in (%scripts%) do (
    set "script_name=%%~a"
    REM Create the shortcut
    echo Creating shortcut for !script_name!...
    call :create_shortcut "!script_name!"
)

echo All shortcuts created.
goto :eof

:create_shortcut (
    set "shortcut_file=%~n1.lnk"
    set "target_path=%~dp0%~1"
    set "working_directory=%cd%"
    set "start_in=%cd%"
    set "script_extension=%~x1"

    echo "!target_path!"
    echo Shortcut file: !shortcut_file!

    if /i "%script_extension%" == ".bat" (
        set "interpreter=cmd.exe"
    ) else if /i "%script_extension%" == ".py" (
        set "interpreter=python.exe"
    ) else if /i "%script_extension%" == ".pl" (
        set "interpreter=perl.exe"
    ) else (
        set "interpreter=cmd.exe"
        echo Unsupported file type: %script_extension%, defaulting to %interpreter%
    )

    for /f "usebackq delims=" %%I in (`where %interpreter%`) do set "interpreter_path=%%I"

    set "powershell_cmd=$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%shortcut_file%'); $Shortcut.TargetPath = '%interpreter_path%'; $Shortcut.Arguments = '$(Resolve-Path -LiteralPath \"%target_path%\")'; $Shortcut.WorkingDirectory = '$(Resolve-Path -LiteralPath \"%working_directory%\")'; $Shortcut.WindowStyle = 1; $Shortcut.Save(); $Shortcut = $WshShell.CreateShortcut('%shortcut_file%'); $Shortcut.WorkingDirectory = '$(Resolve-Path -LiteralPath \"%start_in%\")'; $Shortcut.Save();"
    
    powershell -Command "%powershell_cmd%"
    exit /b
)


:eof
endlocal
