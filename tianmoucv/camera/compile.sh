mkdir build
cd build
DEFAULTPATH="../../../tianmouc_sdk"
echo "please input the Tianmouc_sdk path, the default is: $DEFAULTPATH (enter for defualt)"
read SDK_PATH
if [ -z "$SDK_PATH" ]; then
  echo "Using default name: $DEFAULTPATH"
  SDK_PATH="$DEFAULTPATH"
else
  echo "Using input name: $SDK_PATH"
fi
cmake -DSDK_PATH=$SDK_PATH ..
make
mkdir ../lib
cp *.so ../lib
