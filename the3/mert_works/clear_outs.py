
import os
import sys
import shutil




if __name__ == "__main__":

    path = "./THE3_Images"
    folders = os.listdir(path)

    for folder in folders:
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)