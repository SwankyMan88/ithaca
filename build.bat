@echo off
rem Builds Ithaca (needs MinGW-w64 g++ on PATH). Every exe is static, with no DLLs to ship:
rem   Ithaca.exe        opens Ithaca in your browser (and serves other devices and the API)
rem   Ithaca App.exe    Ithaca in its own window, everything
rem   Ithaca Chat.exe   its own window, chat only
rem   Ithaca Image.exe  its own window, image studio only
rem   Ithaca Video.exe  its own window, video studio only
rem   Ithaca Audio.exe  its own window, speech only
rem Pass "quick" to build only Ithaca.exe.
setlocal enabledelayedexpansion
cd /d "%~dp0"
set FLAGS=-std=c++17 -O3 -mavx2 -mfma -mf16c -DNDEBUG
set LIBS=-lws2_32 -lshell32 -lmfplat -lmfreadwrite -lmfuuid -lole32 -lwinhttp -lwindowscodecs -luuid -lbcrypt -lcrypt32 -ladvapi32 -lgdi32 -luser32
if not exist build\obj mkdir build\obj
if not exist build\main mkdir build\main
echo Compiling...
set OBJS=
for %%f in (src\*.cpp) do (
  if /I not "%%~nf"=="main" (
    g++ %FLAGS% -c %%f -o build\obj\%%~nf.o
    if errorlevel 1 goto failed
    set OBJS=!OBJS! build\obj\%%~nf.o
  )
)
windres src\ithaca.rc -O coff -o build\obj\res.o
if errorlevel 1 goto failed
set OBJS=%OBJS% build\obj\res.o

g++ %FLAGS% -c src\main.cpp -o build\main\main.o
if errorlevel 1 goto failed
g++ %FLAGS% -static -s %OBJS% build\main\main.o -o Ithaca.exe %LIBS%
if errorlevel 1 goto failed
echo   Ithaca.exe
if /I "%~1"=="quick" goto done

for %%e in (App Chat Image Video Audio) do (
  call :lower %%e
  g++ %FLAGS% -DITHACA_EDITION=\"!LOW!\" -c src\main.cpp -o build\main\main_!LOW!.o
  if errorlevel 1 goto failed
  g++ %FLAGS% -static -s -mwindows %OBJS% build\main\main_!LOW!.o -o "Ithaca %%e.exe" %LIBS%
  if errorlevel 1 goto failed
  echo   Ithaca %%e.exe
)
:done
echo Done.
exit /b 0

:lower
set LOW=%~1
for %%a in ("A=a" "B=b" "C=c" "D=d" "E=e" "F=f" "G=g" "H=h" "I=i" "J=j" "K=k" "L=l" "M=m" "N=n" "O=o" "P=p" "Q=q" "R=r" "S=s" "T=t" "U=u" "V=v" "W=w" "X=x" "Y=y" "Z=z") do set LOW=!LOW:%%~a!
exit /b 0

:failed
echo BUILD FAILED
pause
exit /b 1
