#conda install pip
python -m pip install pybind11 -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "compile rdp_pcie lib..."
cd tianmoucv/rdp_pcie/
sh compile_pybind.sh
cd ../..
echo "compile rdp_usb lib..."
cd tianmoucv/rdp_usb/
sh compile_pybind.sh
cd ../..
echo "compile sdk lib..."
cd tianmoucv/camera
sh compile.sh
cd ../..
echo "install requirements..."
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install --no-deps . -i https://pypi.tuna.tsinghua.edu.cn/simple