from PIL import Image, ImageDraw
import random
import os.path
import numpy
import glob
import imageio
from matplotlib import pyplot as plt

class DatasetCreator:

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        backgroundsFolderPath = script_dir +'/Backgrounds'
        picturesFolderPath = script_dir +'/Pics'
        backgroundsPath = glob.glob(backgroundsFolderPath + '/*.jpg')
        picturesPath = glob.glob(picturesFolderPath + '/*.png')
        picturesPath.sort()
        self.savepath = script_dir + "/save_test"

        onlyfiles = [f for f in os.listdir(backgroundsFolderPath) if os.path.isfile(os.path.join(backgroundsFolderPath, f))]

        self.backg_units = [img for img in backgroundsPath]
        self.pic_units = [img for img in picturesPath] 

        self.backg_array = numpy.array( [numpy.array(Image.open(img)) for img in backgroundsPath] )
        self.picture_array = numpy.array([numpy.array(Image.open(img)) for img in picturesPath])

    def randomXY(self, bg, picCrop):
        x1 = random.randint(0, bg.shape[1]-picCrop.size[0]) #picture.shape[1]
        y1 = random.randint(0, bg.shape[0]-picCrop.size[1]) #picture.shape[0]
        x2 = x1 + picCrop.size[0]
        y2 = y1 + picCrop.size[1]
        return [x1, y1, x2, y2]

    def check4tuples(self, check, randomList, bg, picCrop):
        #for j in range(0, len(check)):
        #    while(((check[j][0] <= randomList[2] <= check[j][2]) and (check[j][1] <= randomList[3] <= check[j][3])) or #corner a
        #    ((check[j][0] <= randomList[0] <= check[j][2]) and (check[j][1] <= randomList[1] <= check[j][3])) or #corner c
        #    ((check[j][0] <= randomList[3] <= check[j][2]) and (check[j][1] <= randomList[0] <= check[j][3])) or #corner b
        #    ((check[j][0] <= randomList[1] <= check[j][2]) and (check[j][1] <= randomList[2] <= check[j][3]))): #corner d
        #        print("failed check")
        #        randomList = self.randomXY(bg, picCrop) 
        return randomList

    def write2files(self, x, y, width, height, label, RandInt):
        string = str(label) + " " + str(x) + " " + str(y) + " " + str(width) + " " + str(height) + "\n"
        #print("YOLO txt:")
        #print(string)
        file=open(str(self.savepath) + "/" + "image"  + "-" + str(RandInt) + ".txt", "a")
        #print("write-append")
        file.write(string)
        #print("YOLO txt written")
        file.close()


    def placePicture(self):
        
        RandInt = 0
        bg = []

        for i in range(self.backg_array.shape[0]): #iterate through backgrounds

            RandInt = random.randint(0,1000)
            back_ground = self.backg_units[i]
            check = []
            bg = self.backg_array[i]
            backgr = Image.fromarray(bg, 'RGB')
            piclist = []
            rand = random.randint(1, len(self.pic_units)-1) # pick a random number of ROI-pics e.g. 1 - 4
            print("rand:")
            print(rand)
            for r in range(1, rand): 
                randIndex = random.randint(0, len(self.pic_units) - 1)
                piclist.append(str(self.pic_units[randIndex])) #randomly pick rand number of pics to append to piclist
                # print(str(self.pic_units[random.randint(0, rand)]))

            for f in piclist:
                randIndex = random.randint(0, len(self.pic_units) - 1)
                picture = self.picture_array[randIndex]
                pic = Image.fromarray(picture, 'RGBA')
                picCoor = pic.getbbox()
                picCrop = pic.crop(picCoor)

                [x3, y3, x4, y4] = self.check4tuples(check, self.randomXY(bg, picCrop), bg, picCrop)
                check.append([x3, y3, x4, y4])

                width = float(picCrop.size[0]) / float(bg.shape[1])
                height = float(picCrop.size[1]) / float(bg.shape[0])
                x = float(((float(x4-x3) / 2) + x3) / float(bg.shape[1]))
                y = float(((float(y4-y3) / 2) + y3) / float(bg.shape[0]))

                backgr.paste(picCrop, (x3, y3, x4, y4))
                draw = ImageDraw.Draw(backgr)
                #draw.rectangle([x3, y3, x4, y4], outline=(255,0,0))
                #backgr.show()                    
                self.write2files(x, y, width, height, randIndex, RandInt)
        
            imageio.imwrite(str(self.savepath) + "/" + "image" + "-" + str(RandInt) + ".jpg", backgr)


data = DatasetCreator()
for i in range(10):
    data.placePicture()

