
import os
import sys

if __name__ == "__main__":

    path = "./THE2_Images/Question3/"

    ignore_files = ["c1.jpg", "c2.jpg", "c3.jpg"]

    for filename in os.listdir(path):
        print(filename)
        if (filename.endswith(".jpg") or filename.endswith(".png")) and filename not in ignore_files:
            os.remove(path + filename)