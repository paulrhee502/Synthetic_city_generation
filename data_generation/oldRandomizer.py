import numpy as np
import os

def file_sampler(P=0.85):
    """Takes P between 0 and 1 to select percentage of files to output
        randomly fom larger directory"""
    files = [f for f in os.listdir("/Users/Varun/Documents/CityEngine/Default Workspace/c3/images/sAustin2.3/data/Original_Tiles") if os.path.isfile(os.path.join("/Users/paulrhee/Desktop/Data+/dataset 1/data/asdf",f))]
    print(len(files))
    exclude = [i for i in files if ("austin" in i\
                                 or "vienna" in i\
                                 or "chicago" in i\
                                 #selecting only files with GTs
                                 or "RGB" in i\
                                 or "file" in i\
                                 or "Thumbs" in i\
                                 or "izer" in i)]
    print(len(exclude))
    files = set(files)
    exclude = set(exclude)
    A = files.difference(exclude)
    # select P ratio of the files randomly without replacement
    A = list(A)
    random_files = np.random.choice(A, 500, replace=False)
    with open('files.txt', 'w') as f:
        for item in random_files:
            #saving corresponding color version to GTs
            RGB_item = item[:-6] + "RGB.tif"
            f.write("%s\n" % item)
            f.write("%s\n" % RGB_item)

def make_colTiles(src_dir):
    """Reads a directory for .tif files and makes a randomized list
        of colTileNames.txt"""
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    random_files = np.random.choice(files, len(files), replace=False)
    with open('colTileNames.txt', 'w') as f:
        for file in random_files:
            if "_samplepatch" in file:
                if "GT" in file:
                    name = file[:-19]
                    f.write("%s\n" % name)
                elif "RGB" in file:
                    pass
                    #name = file[:-20]
            elif "GT" in file:
                name = file[:-7]
                f.write("%s\n" % name)

def randomize_colTiles(src_file, dst_file):
    """Will take a colTileNames.txt file as input and randomize the
        order of filenames and write to a new file and makes backup"""
    lineList = [line.rstrip('\n') for line in open(src_file)]
    backup = src_file[:-4] + "_backup.txt"
    with open(backup, 'w') as f:
        for file in lineList:
            f.write("%s\n" % file)
    randomList = np.random.choice(lineList, len(lineList), replace=False)
    os.remove(src_file)
    with open(dst_file, 'w') as f:
        for file in randomList:
            f.write("%s\n" % file)

if __name__ == '__main__':
    file_sampler()

    #randomize_colTiles('/hdd/data+/Paul/FourToOne/meta_data/colTileNames.txt',
    #'/hdd/data+/Paul/FourToOne/meta_data/colTileNames.txt')

    #make_colTiles('/Users/paulrhee/Desktop/Data+/dataset 1/data/FourToOne')
