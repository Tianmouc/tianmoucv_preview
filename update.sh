python -m pip uninstall tianmoucv | echo "y" 
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
ONLY_UPDATE_PYTHON=1 python -m pip install --no-deps . --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple