@echo off
rem Builds Ithaca.exe (needs MinGW-w64 g++ on PATH). Produces one static exe with no DLL dependencies.
cd /d "%~dp0"
g++ -std=c++17 -O3 -mavx2 -mfma -mf16c -DNDEBUG -static -s src\*.cpp -o Ithaca.exe -lws2_32 -lshell32 -lmfplat -lmfreadwrite -lmfuuid -lole32 -lwinhttp -lwindowscodecs
if errorlevel 1 (echo BUILD FAILED & pause & exit /b 1)
echo Built Ithaca.exe
