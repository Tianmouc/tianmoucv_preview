python -m pip install pybind11
echo "compile rod_decode_pybind_usb lib..."
cd tianmoucv/rod_decode_pybind_usb/
sh compile_pybind.sh
cd ../..
python -m pip install .
