import imageio
import numpy as np
import imgaug as ia
import cv2
import os.path
from scipy import misc
import glob
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
ia.seed(1)

def saturate_bbs(l_abs_x1,l_abs_y1,l_abs_x2,l_abs_y2,l_width,l_height):
    
    x_coordinates = np.array([l_abs_x1,l_abs_x2])
    y_coordinates = np.array([l_abs_y1,l_abs_y2])

    x_coordinates[x_coordinates < 0] = 0
    x_coordinates[x_coordinates > l_width] = l_width
    y_coordinates[y_coordinates < 0] = 0
    y_coordinates[y_coordinates > l_height] = l_height

    return x_coordinates[0], y_coordinates[0], x_coordinates[1], y_coordinates[1]


def convert_to_yolo(l_abs_x1,l_abs_y1,l_abs_x2,l_abs_y2,l_width,l_height):

    l_center_x = (float((l_abs_x2 - l_abs_x1))/float(2) + float(l_abs_x1))/float(l_width)
    l_center_y = (float((l_abs_y2 - l_abs_y1))/float(2) + float(l_abs_y1))/float(l_height)
    l_bbWidth = float(l_abs_x2-l_abs_x1)/float(l_width)
    l_bbHeight = float(l_abs_y2-l_abs_y1)/float(l_height)

    return l_center_x,l_center_y,l_bbWidth,l_bbHeight


def convert_from_yolo(l_center_x,l_center_y,l_bbWidth,l_bbHeight,l_width,l_height):
    l_abs_x1 = (l_center_x - l_bbWidth/2)*l_width
    l_abs_x2 = (l_center_x + l_bbWidth/2)*l_width
    l_abs_y1 = (l_center_y + l_bbHeight/2)*l_height
    l_abs_y2 = (l_center_y - l_bbHeight/2)*l_height
    return l_abs_x1,l_abs_x2,l_abs_y1,l_abs_y2

#PATH TO ORIGINAL IMAGES WITH LABELS#
filepathOriginalFolder = "/home/silje/Documents/gitRepos/DatasetCreator/DatasetCreator/createdImages/others/" #script_dir + "/original/"

#AUGMENTED BATCH NAME AND DESIRED MULTIPLE OF ORIGINAL IMAGES#
batchName = "batch18_other_"
desiredMultiple = 5

#PREVIEW N AMOUNT OF BOUNDING BOXES ON AUGMENTED IMAGES#
viewNBoundingBoxes = 0

#DESIRED AUGMENTATION# 
seq = iaa.Sequential([
    #iaa.AddToHue((-50,50)),  # change their color
    #iaa.MultiplySaturation((0.2,1.5)), #calm down color
    #iaa.ElasticTransformation(alpha=10, sigma=6),  # water-like effect (smaller sigma = smaller "waves")
    #iaa.PiecewiseAffine(scale=(0.01,0.05)), #sometimes moves pieces of image around (RAM-heavy)
    #iaa.LogContrast((0.5,1.0),True), #overlay color
    #iaa.MotionBlur(20,(0,288),1,0), #motion blur for realism
    #iaa.BlendAlpha((0.1, 0.7), 
    #iaa.MedianBlur(11), per_channel=True), #alpha-blending with median blur
    #iaa.PerspectiveTransform(scale=(0.01, 0.1)),
    iaa.AdditiveGaussianNoise(scale=0.05*255, per_channel=True), #noise
    #iaa.CoarseDropout(p=0.1, size_percent=0.005), #blocks removed from image
    #iaa.Affine(rotate=(-30,30)), #rotate #PROBLEM WITH BOUNDING BOXES MOSTLY CAUSED BY THIS
    #iaa.Fliplr(0.5)
], random_order=True)

#---------------------------------------------------------------------

class DatasetAugmentor:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath_img = script_dir + "/augmented/" + batchName + "%d.jpg" #
        self.filepath_txt = script_dir + "/augmented/" + batchName + "%d.txt" #

        self.imagesToAugment = []
        self.bbs_images = []
        self.augmentedImages = []
        self.augmented_bbs = []

        originalImagesList = glob.glob(filepathOriginalFolder+"*.jpg")
        originalImagesList.sort()
        self.imageList = originalImagesList

        originalLabelsList = glob.glob(filepathOriginalFolder+"*.txt")
        originalLabelsList.sort()
        self.labelList = originalLabelsList

        self.sequential = seq

    def loadBoundingBoxes(self, imageList, labelList, imagesToAugment):

        for i in range(len(imageList)):

            f = open(labelList[i], "r")
            contentsOfFile = f.readlines()
            for u in range(len(contentsOfFile)):
                contentsOfFile[u] = contentsOfFile[u].split()
                for j in range(len(contentsOfFile[u])):
                    contentsOfFile[u][j] = float(contentsOfFile[u][j])

            image_height, image_width = imagesToAugment[i].shape[:2]

            boundingBoxes = []
            for u in range (len(contentsOfFile)):
                abs_x1,abs_x2,abs_y1,abs_y2 = convert_from_yolo(contentsOfFile[u][1],contentsOfFile[u][2],contentsOfFile[u][3],contentsOfFile[u][4],image_width,image_height)
                bb = BoundingBox(x1=abs_x1, x2=abs_x2, y1=abs_y1, y2=abs_y2, label = int(contentsOfFile[u][0]))
                boundingBoxes.append(bb)

            bboi = BoundingBoxesOnImage(boundingBoxes, shape=imagesToAugment[i].shape)
            self.bbs_images.append(bboi)

            f.close()

    def readAndAppendImages(self, originalImageList):
        for i in range(len(originalImageList)):
            img = imageio.imread(originalImageList[i])
            self.imagesToAugment.append(img)

    def createMultipleBatches(self,imagesToAugment,bbs_images):
        self.imagesToAugment= imagesToAugment*desiredMultiple
        self.bbs_images = bbs_images*desiredMultiple
    
    def augmentImages(self,imagesToAugment,bbs_images):
        self.augmentedImages, self.augmented_bbs = self.sequential(images=imagesToAugment, bounding_boxes=bbs_images)

    def saturateBoundingBoxes(self, augmentedImages, augmented_bbs):
        for i, image in enumerate(augmentedImages):
            heightImage, widthImage = image.shape[:2]
            widthImage = float(widthImage)
            heightImage = float(heightImage)
            for u in range(len(augmented_bbs[i].bounding_boxes)):
                bb_info = augmented_bbs[i].bounding_boxes[u]
                if (not bb_info.is_fully_within_image(image)):
                    abs_x1,abs_y1,abs_x2,abs_y2 = saturate_bbs(bb_info.x1_int,bb_info.y1_int,bb_info.x2_int,bb_info.y2_int,widthImage,heightImage)
                    augmented_bbs[i].bounding_boxes[u] = BoundingBox(x1=abs_x1, x2=abs_x2, y1=abs_y1, y2=abs_y2, label = bb_info.label)

    def viewPreviewImages(self,images_aug,bbs_aug):
        for i in range(viewNBoundingBoxes):
            ia.imshow(bbs_aug[i].draw_on_image(images_aug[i], size=5))

    def saveImagesAndLabels(self,images_aug,bbs_aug, filepath_img, filepath_txt):
        for i, image_aug in enumerate(images_aug):
            f=open(filepath_txt % i,"w")

            heightImage, widthImage = image_aug.shape[:2]
            widthImage = float(widthImage)
            heightImage = float(heightImage)

            for u in range(len(bbs_aug[i].bounding_boxes)):
                bb_info = bbs_aug[i].bounding_boxes[u]
                center_x,center_y,bbWidth,bbHeight = convert_to_yolo(bb_info.x1_int,bb_info.y1_int,bb_info.x2_int,bb_info.y2_int,widthImage,heightImage)
                f.write(str(bb_info.label) + ' ' + str(round(center_x,6)) + ' ' + str(round(center_y,6)) + ' ' + str(round(float(bbWidth),6)) + ' ' + str(round(float(bbHeight),6)) + '\n')
                
            misc.imsave(filepath_img % i, image_aug)
            f.close()

    def createAugmentedSet(self):

        self.readAndAppendImages(self.imageList)
        self.loadBoundingBoxes(self.imageList, self.labelList, self.imagesToAugment)
        self.createMultipleBatches(self.imagesToAugment,self.bbs_images)
        self.augmentImages(self.imagesToAugment,self.bbs_images)
        self.saturateBoundingBoxes(self.augmentedImages,self.augmented_bbs)
        
        if (viewNBoundingBoxes > 0):
            self.viewPreviewImages(self.augmentedImages,self.augmented_bbs)

        self.saveImagesAndLabels(self.augmentedImages,self.augmented_bbs, self.filepath_img, self.filepath_txt)


augment = DatasetAugmentor()
augment.createAugmentedSet()

