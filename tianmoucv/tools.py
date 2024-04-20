import numpy as np
import requests
from urllib.parse import urlparse
import tempfile
import os 

USB_MODULE_MAT = np.array([[462.15896054,   0.          ,307.03359738],
                     [  0.         ,462.40462068   ,149.10361709],
                     [  0.         ,  0.           ,1.        ]])


def check_url_or_local_path(input_str):
    parsed_url = urlparse(input_str)
    if parsed_url.scheme in ['http', 'https']:
        return 1
    elif parsed_url.scheme == '' and parsed_url.path:
        return 0
    else:
        return -1

def download_file(url,file_name=None):
    temp_dir = tempfile.gettempdir()
    if file_name is None:
        file_name = './nn.ckpt'
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f'File downloaded successfully as {file_name}')
        return file_name
    else:
        print('Failed to download the file')
        return None