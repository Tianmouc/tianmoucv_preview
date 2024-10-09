from setuptools import find_packages
from setuptools import setup

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='tianmoucv',                     # 模块的名称
    version='0.3.3.2',                      # 版本号
    author='Yihan Lin,Taoyi Wang',        # 作者名称
    author_email='532109881@qq.com',      # 作者邮箱
    description='Algorithms library for Tianmouc sensor',   # 简要描述
    url='https://github.com/Tianmouc/tianmoucv',  # 项目主页的URL
    packages=find_packages(),   # 告诉 setuptools 自动找到要安装的包
    package_data = {'':['data/blc/*','rdp_usb/*','rdp_pcie/*']},
    include_package_data=True,
    install_requires=install_requires,
    long_description=long_description,
    # 可选的内容
    keywords='tianmoucv',           # 关键词
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
    ],
    python_requires='>=3.8'
)
