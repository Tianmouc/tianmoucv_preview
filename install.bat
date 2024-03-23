@echo off

REM Install pybind11
::conda install pip
python -m pip install pybind11

echo "compile rod_decode_pybind_usb lib..."
cd tianmoucv\rod_decode_pybind_usb
call compile_pybind.bat
cd ..\..

REM Install Python package
python -m pip install .