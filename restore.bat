@echo off
setlocal enabledelayedexpansion enableextensions

call :full_path backup_dir "__backup"

set /p "confirm=Do you want to restore backups? (Y/N) "

if /i "%confirm%"=="y" (
    if exist "%backup_dir%" (
        for /r "%backup_dir%" %%d in (*.par) do (
            set "backup_path=%%~fd"
            set "backup_name=%%~nd"

            set "file_path=!backup_path:%backup_dir%=%cd%!"

            copy /y "%%d" "!file_path!"
            echo "Restored !file_path!"
        )

        echo "Backup files restored to current directory."
    )
)
goto :eof

:full_path <resultVar> <pathVar> (
    set "%~1=%~f2"
    exit /b
)

:eof
endlocal
