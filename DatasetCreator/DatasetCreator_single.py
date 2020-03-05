from PIL import Image, ImageDraw
import random
import os.path
import numpy
import glob
from pathlib import Path
from matplotlib import pyplot as plt

# CONDITIONS FOR USE:
# - all pictures in pictures folder are .png images, names with their corresponding label as the first character in the filename. Example: 0.png, 0 (copy).png, 00.png will all have the same class 0.
# - pictures in backgrounds folder are .jpg and larger than all the .png images
# - change parameters marked with "////"

#//// change lowest and highest scaling of image
LOWEST_SCALE = 0.05
HIGHEST_SCALE = 1.0

#//// View images?
VIEW_PREVIEW_IMAGES = False

#//// change amount of sets desired
TOTAL_SET_NUMBER = 2


class DatasetCreator:

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        backgroundsFolderPath = script_dir +'/backgrounds'
        picturesFolderPath = script_dir +'/subjectPics'
        backgroundsPath = glob.glob(backgroundsFolderPath + '/*.jpg')
        picturesPathList = glob.glob(picturesFolderPath + '/*.png')
        self.savepath = script_dir + "/createdImages"

        self.labelList = []
        for i in range(len(picturesPathList)):
            self.labelList.append(Path(picturesPathList[i]).stem[0])

        #onlyfiles = [f for f in os.listdir(backgroundsFolderPath) if os.path.isfile(os.path.join(backgroundsFolderPath, f))]

        self.backg_array = numpy.array( [numpy.array(Image.open(img)) for img in backgroundsPath] )
        self.picture_array = numpy.array([numpy.array(Image.open(img)) for img in picturesPathList])

    def placePictureRandom(self, label, background, picture, i, j):

        pic = Image.fromarray(picture, 'RGBA')

        randomScalingFactor = random.uniform(LOWEST_SCALE,HIGHEST_SCALE)
        pic = pic.resize((int(randomScalingFactor*pic.width),int(randomScalingFactor*pic.height)), Image.ANTIALIAS)

        picCoor = pic.getbbox()
        picCrop = pic.crop(picCoor)

        x1 = random.randint(0, background.shape[1]-picCrop.size[0]) #picture.shape[1]
        y1 = random.randint(0, background.shape[0]-picCrop.size[1]) #picture.shape[0]
        x2 = x1 + picCrop.size[0]
        y2 = y1 + picCrop.size[1]

        # Normalizing for YOLO-label format
        width = float(picCrop.size[0]) / float(background.shape[1])
        height = float(picCrop.size[1]) / float(background.shape[0])
    
        x = (((x2-x1) / 2) + x1) / float(background.shape[1])
        y = (((y2-y1) / 2) + y1) / float(background.shape[0])

        backg = Image.fromarray(background, 'RGB')
        
        backg.paste(picCrop, (x1, y1), picCrop)
        draw = ImageDraw.Draw(backg)

        if (VIEW_PREVIEW_IMAGES == True):
            draw.rectangle([x1,y1,x2,y2], outline=(255,0,0))
            #draw.line([((x-width/2)*background.shape[1],y*background.shape[0]), ((x+width/2)*background.shape[1],y*background.shape[0])], fill=(0,0,255), width = 3)
            #draw.line([(x*background.shape[1],(y-height/2)*background.shape[0]), (x*background.shape[1],(y+height/2)*background.shape[0])], fill=(0,0,255), width = 3)
            backg.show()
        
        
        self.save_to_folder(label,x,y,width,height,backg,i,j)



    def save_to_folder(self,label,x,y,width,height,image,i, j):
        string = str(label) + " " + str(x) + " " + str(y) + " " + str(width) + " " + str(height)
        randomInt = str(random.randint(0,1000))
        file=open(self.savepath + "/" + "image" + str(i+10) + "-" + str(j) + "-" + randomInt + ".txt", "w")
        file.write(string)
        file.close()
        image.save(self.savepath + "/" + "image" + str(i+10) + "-" + str(j) + "-" + randomInt + ".jpg")  
        
  

    def createEntireSet(self):
        #print(self.picture_array.shape)
        for i in range(self.picture_array.shape[0]):
            for j in range(self.backg_array.shape[0]):
                self.placePictureRandom(self.labelList[i], self.backg_array[j], self.picture_array[i], i, j)


data = DatasetCreator()

for i in range(TOTAL_SET_NUMBER):
    data.createEntireSet()

