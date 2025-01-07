import os.path
import logging
import requests as rq

logging.basicConfig(level=logging.INFO)

def download(url:str, file_path:str)->None:
    if os.path.isfile(file_path):
        logging.info("File already exists!")
    else:
        logging.info("Downloading file ...")
        response = rq.get(url)
        if response.status_code == 200:
            with open(file_path, "w") as file_conn:
                file_conn.write(response.text)