from PIL import Image, ImageFile
from os import listdir
from os.path import isfile, join, splitext
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

def randomCrop(dir_path, save_dir, crop_w, crop_h):
    """
    @param dir_path: The path to the directory with imgaes from Inria dataset (5000x5000)
    @param save_dir: Path to the directory to which the cropped images will be saved
    @param crop_w, crop_h: Desired width/height respectively after crop
    return: cropped images in save_dir 
    """
    corners = {} #dictionary that holds the randomly selected corners for each tile so RGB and GT use the same corner
    for fil in [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and splitext(f)[1] == ".tif"]:
        print(fil)
        image_obj = Image.open(join(dir_path, fil))
        original_w, original_h = image_obj.size
        print("ORIGINAL SIZE: " + str(original_w) + " " + str(original_h))
        if(original_w < crop_w or original_h < crop_h):
            print("Desired cropped image dimensions are larger than the original image")
            continue
        tile = fil.split("_")[0]
        if(tile not in corners):
            corners[tile] = (random.randint(0,original_w - crop_w), random.randint(0,original_h - crop_h))
        cropped_image = image_obj.crop((corners[tile][0], corners[tile][1], corners[tile][0] + crop_w, corners[tile][1] + crop_h))
        cropped_image.save(join(save_dir, splitext(fil)[0] + ".tif"))

def cropInriaOverlap(dir_path, save_dir):
    """
    @param dir_path: The path to the directory with imgaes from Inria dataset (5000x5000)
    @param save_dir: Path to the directory to which the cropped images will be saved
    """
    for fil in [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and splitext(f)[1] == ".tif"]:
        print(fil)
        image_obj = Image.open(join(dir_path, fil))
        splitIndex = fil.rfind('_')
        extIndex = fil.rfind('.')
        for x in range(10):
            for y in range(10):
                index = (10 * x) + y
                cropped_image = image_obj.crop((492 * x, 492 * y, (492 * x) + 572, (492 * y) + 572)) #Produces a 572x572 image from cropping 5000x5000 inria image
                cropped_image.save(join(save_dir, fil[0:splitIndex] + "_" + str(index) + fil[splitIndex:extIndex] + ".tif"))

if __name__ == '__main__':
    #cropInriaOverlap("/hdd/data+/Source/inria/data/Original_Tiles", "/hdd/data+/Source/inria_cropped_tif/data/Original_Tiles")
    randomCrop('/Users/paulrhee/Desktop/example_source/', '/Users/paulrhee/Desktop/example_dest/', 572, 572)