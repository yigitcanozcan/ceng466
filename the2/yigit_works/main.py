


import os
import subprocess
import sys


if __name__ == "__main__":
    run_files = ["the2.1.py", "the2.2.py", "the2.3.py"]
    print("LOG: Outputs writtten to folders in ./THE2_Images/")
    for filename in run_files:
        print(filename)
        subprocess.call([sys.executable, filename])