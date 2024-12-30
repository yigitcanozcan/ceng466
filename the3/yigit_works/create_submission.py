

import zipfile

files = ["the3_report.pdf", "the3_solution.py"]


with zipfile.ZipFile("submission.zip", "w") as zip:
    for file in files:
        zip.write(file)