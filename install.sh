python -m pip uninstall tianmoucv | echo "y" 
python -m pip install pybind11 -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "compile rdp_usb lib..."
python -m pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
