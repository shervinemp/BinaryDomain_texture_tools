@REM Move files/dirs passed as input between '__staged' and '__unstaged'.
@REM
@REM Usage:
@REM stash.bat <file/dir> [<file/dir> ...]
@REM
@REM Example:
@REM stash.bat file1 file2 dir1
@REM
@REM This will move file1, file2, and dir1 from '__staged' to '__unstaged', or vice versa.
@REM
@REM Note:
@REM If the file/dir is not under '__staged' or '__unstaged', it will be ignored.
@REM

@echo off
setlocal enabledelayedexpansion enableextensions

set "staged=__staged\"
set "unstaged=__unstaged\"

set "files=%*"
for %%i in (%files%) do (
    set "file=%%~fi"
    if "!file:~-1!"=="\" set "file=!file:~0,-1!"

    set "wo_mod=!file:%staged%=!"
    set "wo_usg=!file:%unstaged%=!"

    if not "!wo_mod!"=="!%file!" (
        set "file_dest=!file:%staged%=%unstaged%!"
    ) else if not "!wo_usg!"=="!file!" (
        set "file_dest=!file:%unstaged%=%staged%!"
    ) else (
        echo "!file! is not under '__staged' or '__unstaged'."
        exit /b
    )

    if exist "%file_dest%" (
        set /p "confirm=!file_dest! already exists."
        exit /b
    )

    for %%j in ("!file_dest!") do set "dir_dest=%%~dpj"
    mkdir "!dir_dest!" 2>nul
    if exist "!file!\" (
        robocopy "!file!" "!file_dest!" /move
    ) else (
        for %%j in ("!file!") do (
            set "dir_source=%%~dpj"
            set "file=%%~nxj"
        )
        robocopy "!dir_source:~0,-1!" "!dir_dest:~0,-1!" "!file!" /mov
    )
)