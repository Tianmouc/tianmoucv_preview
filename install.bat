@echo off

REM Install pybind11
::conda install pip
python -m pip install pybind11

echo "compile rdp_pcie lib..."
cd tianmoucv\rdp_pcie
call compile_pybind.bat
cd ..\..

echo "compile rdp_usb lib..."
cd tianmoucv\rdp_usb
call compile_pybind.bat
cd ..\..

REM Install Python package
python -m pip install .