import csv
import os
from shutil import copy2

with open('awe_dataset.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for (db, file, id) in reader:
        original_path = f"awe/{id.zfill(3)}/{file}"
        annotations_path = f"awe/{id.zfill(3)}/annotations.json"
        if not os.path.exists(f"data/{db}/{id.zfill(3)}"):
            os.mkdir(f"data/{db.zfill(3)}/{id.zfill(3)}")
        print(original_path)
        copy2(original_path, f"data/{db}/{id.zfill(3)}/{id}_{file}")
        copy2(annotations_path, f"data/{db.zfill(3)}/{id.zfill(3)}/annotations.json")
