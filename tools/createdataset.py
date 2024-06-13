import os, random, shutil, math, re
FILECAP = 200000

def create_dataset(path_data:str, path_tvt, tvt_split:(float,float,float) = (0.6, 0.2, 0.2), fsplit=(0,0), seed=0):
    '''Generates the train/validation/test folders from the data at path_tvt from data provided by path_data with a ratio 
    specified by tvt_split. seed shuffles the data beforehand if != 0 with that seed and fsplit switches create_dataset into kfold mode if != (0,0).
    fsplit[1] specifies the general number of folds while fsplit[0] is the current fold. This is a very inefficient variant of kfold and
    should be replaced with a map pointing to the respective files'''
    i = 0

    try:
        shutil.rmtree(path_tvt+"train")
        shutil.rmtree(path_tvt+"test")
        shutil.rmtree(path_tvt+"val")
    except:
        pass
    try:
        os.mkdir(path_tvt+"train")
        os.mkdir(path_tvt+"test")
        os.mkdir(path_tvt+"val")
    except:
        pass

    for root, dirs, files in os.walk(path_data):
        print(i)
        if not i:
            for dir in dirs:
                try:
                    os.mkdir(path_tvt+"train\\"+dir)
                    os.mkdir(path_tvt+"test\\"+dir)
                    os.mkdir(path_tvt+"val\\"+dir)
                except:
                    continue
            dir_list = dirs.copy()
            i += 1
            continue
        if seed != 0: 
            random.Random(seed).shuffle(files) 
        if fsplit==(0,0):
            for c, file in enumerate(files[:FILECAP]):
                if c < math.floor(len(files)*tvt_split[0]):
                    shutil.copyfile(root+"\\"+file, path_tvt+"train\\"+dir_list[i-1]+"\\"+file)
                elif c < math.floor(len(files)*tvt_split[0])+math.floor(len(files)*tvt_split[1]):
                    shutil.copyfile(root+"\\"+file, path_tvt+"val\\"+dir_list[i-1]+"\\"+file)
                else:
                    if not tvt_split[2] == int(0):
                        shutil.copyfile(root+"\\"+file, path_tvt+"test\\"+dir_list[i-1]+"\\"+file)
        else:
        
#            indexmap = [re.match("(?P<first>\d+)" , s).group('first') for s in files[:FILECAP]]
            for c, file in enumerate(files[:FILECAP]):
                if c > math.floor((len(files)/fsplit[1]) * fsplit[0]) and c < math.floor((len(files)/fsplit[1]) * (1+fsplit[0])):
                    shutil.copyfile(root+"\\"+file, path_tvt+"val\\"+dir_list[i-1]+"\\"+file)
                else:
                    shutil.copyfile(root+"\\"+file, path_tvt+"train\\"+dir_list[i-1]+"\\"+file)
            
        i += 1