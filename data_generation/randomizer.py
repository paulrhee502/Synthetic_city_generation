"""
Author: Varun Nair
Date: 6/7/2019
"""

import numpy as np
import os

def file_sampler(P=0.1, filter=0):
    """Takes P between 0 and 1 to select percentage of files to output
        randomly fom larger directory"""
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    always_exclude = [i for i in files if ("RGB" in i\
                                 or "file" in i\
                                 or "Thumbs" in i\
                                 or "izer" in i)]
    files = set(files)
    always_exclude = set(always_exclude)
    A = files.difference(always_exclude)

    if filter == 0:
        exclude = [i for i in files if ("austin" in i\
                                     or "vienna" in i\
                                     or "chicago" in i\
                                     or "Austin" in i\
                                     or "Vienna" in i\
                                     or "Chicago" in i\
                                     or "kitsap" in i)]
        exclude = set(exclude)
        A = A.difference(exclude)
    elif filter != 0:
        include = [i for i in files if (filter in i)]
        include = set(include)
        A = A.intersection(include)

    A = list(A)
    random_files = np.random.choice(A, 250, replace=False)
    with open('files.txt', 'w') as f:
        for item in random_files:
            #saving corresponding color version to GTs
            RGB_item = item[:-6] + "RGB.tif"
            f.write("%s\n" % item)
            f.write("%s\n" % RGB_item)

def make_colTiles():
    """Reads a directory of tiles and makes colTileNames.txt"""
    files = [f for f in os.listdir('.') if (os.path.isfile(f) and 'GT' in f)]
    random_files = np.random.choice(files, 500, replace=False)
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

def make_GTlist():
    files = [f for f in os.listdir('.') if (os.path.isfile(f) and 'GT' in f)]
    with open('files.txt', 'w') as f:
        for file in files:
            f.write(str(file) + '\n')

if __name__ == '__main__':
    #file_sampler()

    #randomize_colTiles('/hdd/data+/Source/testing_set/meta_data/colTileNames.txt',\
    #'/hdd/data+/Source/testing_set/meta_data/colTileNames.txt')
    #make_GTlist()
    make_colTiles()
