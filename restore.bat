@echo off
setlocal enabledelayedexpansion enableextensions

call :full_path backup_dir "backup_"

set /p "confirm=Do you want to restore backups? (Y/N) "

if /i "%confirm%"=="y" (
    set /p "confirm_all=Do you want to restore all (Y) or one at a time (N)? "
    set "confirm_one=y"

    if exist "%backup_dir%" (
        for /r "%backup_dir%" %%d in (*.par) do (
            set "backup_path=%%~fd"
            set "backup_name=%%~nd"

            if /i "%confirm_all%"=="n" (
                set /p "confirm_one=Do you want to restore !backup_name!? (Y/N) "
            )

            if /i "%confirm_one%"=="y" (
                set "file_path=!backup_path:%backup_dir%=%cd%!"

                copy /y "%%d" "!file_path!"
                echo "Restored !file_path!"
            )
        )

        echo "Backup files restored to current directory."

        if /i "%confirm_all%"=="y" (
            set /p "confirm_delete=Do you want to delete the backup directory? (Y/N) "
        )
        if /i "%confirm_delete%"=="y" (
            rd /s /q "%backup_dir%"
            echo "Deleted backup directory %backup_dir%."
        )
    )
)
goto :eof

:full_path <resultVar> <pathVar> (
    set "%~1=%~f2"
    exit /b
)

:eof
endlocal
