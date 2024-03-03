python -m pip install pybind11
echo "compile rod_decode_pybind lib..."
cd tianmoucv/rod_decode_pybind/
sh compile_pybind.sh
cd ../..
echo "compile rod_decode_pybind_usb lib..."
cd tianmoucv/rod_decode_pybind_usb/
sh compile_pybind.sh
cd ../..
echo "compile sdk lib..."
cd tianmoucv/camera
sh compile.sh
cd ../..
python -m pip install .
