import os 

path = "dataset_without_background/"
input = os.listdir(path)

files = []

for x in input:
    files += [(int(x[x.index("_",0)+1:x.index("_",6)]), x)]


    
files.sort(key=lambda x: x[0])
for x in files: 
    print(x[1])