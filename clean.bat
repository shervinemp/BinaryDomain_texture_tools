@echo off
setlocal enabledelayedexpansion enableextensions

call :full_path staged_dir "__staged"
call :full_path unstaged_dir "__unstaged"

call "%~dp0\restore.bat"

set /p "confirm=Do you want to remove the extracted directories? (Y/N) "

if /i "%confirm%"=="y" (
    for /d /r "%cd%" %%G in (*.par) do (
        if /i not "%%~fG"=="%cd%\%staged_dir%" (
            if /i not "%%~fG"=="%cd%\%unstaged_dir%" (
                echo Removing directory "%%G" ...
                rd "%%G" /s /q 2>nul
            )
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
