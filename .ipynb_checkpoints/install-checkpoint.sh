python -m pip install pybind11
echo "compile rdp_usb lib..."
cd tianmoucv/rdp_usb/
sh compile_pybind.sh
cd ../..
python -m pip install .
