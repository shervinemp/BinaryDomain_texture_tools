@echo off
setlocal enabledelayedexpansion enableextensions

call "%~dp0\restore.bat"

set /p "confirm=Do you want to remove the extracted directories? (Y/N) "

if /i "%confirm%"=="y" (
    
    for /d /r "%cd%" %%G in (*.par) do (
        @REM expand remove %cd% from path
        set "full_path=%%~fG"
        set "relative_path=!full_path:%cd%\=!"
        if not "!relative_path:~0,2!" == "__" (
            echo Removing directory "!relative_path!"...
            rd "%%G" /s /q 2>nul
        )
    )

    echo All directories with suffix ".par" have been removed.

) else (
    echo Operation canceled by user.
)
goto :eof

:full_path <resultVar> <pathVar> (
    set "%~1=%~f2"
    exit /b
)

:eof
endlocal
