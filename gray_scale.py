from PIL import Image
import numpy as np
#import sys
import os
import csv



format='.jpg','jpeg' #Verilerin uzantıları
myDir = "/home/gorkem/PycharmProjects/emotion/" #Verilerin yolu
fileList = []
print(myDir)
for root, dirs, files in os.walk(myDir, topdown=False):
    for name in files:
        if name.endswith(format):
            fullName = os.path.join(root, name)
            fileList.append(fullName)



for file in fileList:
    print(file)
    img_file = Image.open(file)
    #img_file.show()

    # orjinal verilerin parametrelerini al...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # veriyi grileştir
    img_grey = img_file.convert('L')
    img_grey.save('result.png')
    #img_grey.show()

    # grileştirilmiş veriyi kaydet
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    print(value)
    with open("datas.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)