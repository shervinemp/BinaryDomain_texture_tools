@echo off
setlocal enabledelayedexpansion enableextensions

set "par_tool=ParTool.exe"
set "par_tool_args=extract"

call :full_path modified_dir "modified_"

set /p "confirm=Do you want to make a mirror structure under the modified directory? (Y/N) "

call :process_dir "%cd%"
goto :eof

:process_dir (
    for /r "%~1" %%F in (*.par) do (
        call :process_file "%%~fF"
    )

    for /d %%D in ("%~1\*") do (
        if not "%%~fD"=="%cd%" call :process_dir "%%~fD"
    )
    exit /b
)

:process_file (
    set "file_path=%~1"
    set "archive_name=%~n1"

    set "output_dir=!file_path!_"
    set "outputm_dir=!file_path:%cd%=%modified_dir%!_"

    for %%I in (!pathToCheck!) do set parentDir=%~dpI
    if not exist "!parentDir!" mkdir "!parentDir!" 2>nul
    if /i "%confirm%"=="y" (
        mkdir "!outputm_dir!" 2>nul
    )

    "%par_tool%" %par_tool_args% "!file_path!" "!output_dir!"

    for /r "!output_dir!" %%G in (*.par) do (
        call :process_file "%%~fG" "!output_dir!"
    )
    exit /b
)


:full_path <resultVar> <pathVar> (
    set "%~1=%~f2"
    exit /b
)

:eof
endlocal