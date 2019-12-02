from PIL import Image

def crop(imagePath, cor = (2500,0), newSize=(2000,2000)):
    """Defines a function to crop a single segment out of an image
    @param: imagePath specifies image to crop
    @param cor: upper left corner to start crop from
    @param newSize: tuple of new image dimensions
    """

    im = Image.open(imagePath)
    #crops sides as (left, top, right, bottom)
    image = im.crop((cor[0], cor[1], cor[0] + newSize[0], cor[1] + newSize[1]))
    image.save(imagePath.replace('.jpg','_crop.jpg'))

if __name__ == '__main__':
    crop('/Users/Varun/Documents/CityEngine/Default Workspace/c3/images/accuracy/R0+syn/vienna1_RGB.jpg')
