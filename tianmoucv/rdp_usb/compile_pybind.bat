REM Get the active Anaconda environment path
set "ANACONDA_BASE=%CONDA_PREFIX%"

REM Set the paths
set "PYBIND11_INCLUDE=%ANACONDA_BASE%\Lib\site-packages\pybind11\include"
set "PYTHON_INCLUDE=%ANACONDA_BASE%\include"
set "PYTHON_LIBS=%ANACONDA_BASE%\libs"

set "pattern=Python3*.dll"
set "result="

for /r %ANACONDA_BASE% %%a in (%pattern%) do (
    set "fileName=%%~nxa"
    echo %fileName%
    if /I not "%%~nxa"=="python3.dll" (
        set "result=%%~dpnxa"
    )
)


echo The files are: %result%
for %%F in ("%result%") do set "libname=%%~nF"
echo %libname%
echo %PYBIND11_INCLUDE%
echo %PYTHON_INCLUDE%
echo %PYTHON_LIBS%

REM Compile the project with g++
g++ -std=c++20 -O2 .\rod_decoder_py.cpp -shared -o rod_decoder_py.pyd -I"%PYBIND11_INCLUDE%" -I"%PYTHON_INCLUDE%" -L"%PYTHON_LIBS%" -l%libname%  -static


