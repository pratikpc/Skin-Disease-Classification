import os
import csv
import shutil

image_list = dict()
labels = os.listdir("data")
future = []

for images in labels:
        sub_dir = os.path.join("data",images)
        image = os.listdir(sub_dir)
        image_list[images] = image
keys = list(image_list.keys())

original_csv = open('HAM10000_metadata.csv','r') 
data = csv.DictReader(original_csv)

i = 0
for x in data:
        if i == 0:
                fields = list(x) 
                break

with open('test.csv','w', newline='') as test_csv:
        write = csv.DictWriter(test_csv, fieldnames = fields)
        for row in data:
                if i == 0:
                        write.writeheader()
                        i = i + 1
                if row['image_id'] + ".jpg" in image_list[row['dx']]:
                        write.writerow(row)

original_csv.close() 

for key in keys:
        directory = os.path.join("data",key)
        images = os.listdir(directory)
        for image in images:
                image = os.path.join(directory, image)
                shutil.copy(image, "data")
        shutil.rmtree(directory)

                
                
        
