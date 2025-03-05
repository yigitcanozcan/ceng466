
# CENG 466 THE2
# Mert Uludoğan 2380996
# Yiğitcan Özcan 2521847

import os
import sys

if __name__ == "__main__":

    path = "./THE2_Images/Question2/"

    ignore_files = ["b1.jpg", "b2.jpg", "b3.jpg"]

    for filename in os.listdir(path):
        print(filename)
        if (filename.endswith(".jpg") or filename.endswith(".png")) and filename not in ignore_files:
            os.remove(path + filename)