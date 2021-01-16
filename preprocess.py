import csv
from shutil import copy2

data = "/awe"
with open('awe-translation_1.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for (db, file, id) in reader:
        original_path = f"awe/{id.zfill(3)}/{file}"
        print(original_path)
        copy2(original_path, f"data/{db}/{id}_{file}")
