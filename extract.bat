@echo off
setlocal enabledelayedexpansion enableextensions

set "par_tool=ParTool.exe"
set "par_tool_args=extract"

call :full_path staged_dir "__staged"

set /p "confirm=Do you want to make a mirrored directory hierarchy under __staged for staged changes? (Y/N) "

call :process_dir "%cd%"
goto :eof

:process_dir (
    for /r "%~1" %%F in (*.par) do (
        call :process_file "%%F"
    )
    exit /b
)

:process_file (
    set "file_path=%~f1"
    echo !file_path:%cd%=.!
    
    set "dir_name=%~nx1"
    set "output_dir=!file_path:%dir_name%=_%dir_name%!"
    if exist "!output_dir!" (
        set /p "confirm=!output_dir! already exists. Continuing this operation will delete the existing directory. Do you want to continue? (Y/N) "
        if /i "%confirm%"=="n" (
            exit /b
        )
        rmdir /s /q "!output_dir!"
    )

    set "outputm_dir=!output_dir:%cd%=%staged_dir%!"
    if /i "%confirm%"=="y" (
        mkdir "!outputm_dir!" 2>nul
    )

    "%par_tool%" %par_tool_args% "!file_path!" "!output_dir!"

    exit /b
)


:full_path <resultVar> <pathVar> (
    set "%~1=%~f2"
    exit /b
)

:eof
endlocal