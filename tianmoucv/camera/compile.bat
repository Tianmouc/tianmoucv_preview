@echo off
set "DEFAULTPATH=J:/projects/tianmoucv_windows_dependency/tianmouc_sdk"
set /p SDK_PATH="please input the Tianmouc_sdk path, the default is: %DEFAULTPATH% (press Enter for default): "
if "%SDK_PATH%"=="" (
  echo Using default name: %DEFAULTPATH%
  set "SDK_PATH=%DEFAULTPATH%"
) else (
  echo Using input name: %SDK_PATH%
)
cd build
cmake -G "MinGW Makefiles" -DSDK_PATH=%SDK_PATH%  ..
mingw32-make
mkdir ..\lib
copy *.dll ..\lib