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


script_dir = os.path.dirname(os.path.abspath(__file__))

#----------------------------------------------------------
#(PRE)CONDITIONS FOR (SUCCESSFUL) USE:
# - you have .jpg files containing one labeled object per image.
# - labels are formatted in the standard YOLOv3 format, being: bb_class   bb_x_center   bb_y_center   bb_width   bb_height
# - you have not gone too crazy with the amount of augmenters at once
# - your paths are correct
#-----------------------------------------------------------

#PATH TO ORIGINAL IMAGES WITH LABELS#
filepathOriginalFolder = "/home/silje/Documents/computer_vision_vortex/DatasetCreator/createdImages/" #script_dir + "/original/"

#AUGMENTED BATCH NAME AND DESIRED MULTIPLE OF ORIGINAL IMAGES#
batchName = "testBatch2_"
desiredMultiple = 1

#PREVIEW N AMOUNT OF BOUNDING BOXES ON AUGMENTED IMAGES#
viewNBoundingBoxes = 0

#SAVE AUGMENTED IMAGES WITH BOUNDING BOX TO FILEPATH#
filepathSaveFolder = script_dir + "/augmented/"

filepath_img = filepathSaveFolder + batchName + "%d.jpg"
filepath_txt = filepathSaveFolder + batchName + "%d.txt"

#DESIRED AUGMENTATION# 
seq = iaa.Sequential([
    #iaa.AddToHue((-255,255)),  # change their color
    iaa.MultiplySaturation((0.1,0.7)), #calm down color
    #iaa.ElasticTransformation(alpha=20, sigma=4),  # water-like effect (smaller sigma = smaller "waves")
    #iaa.PiecewiseAffine(scale=(0.01,0.05)), #sometimes moves pieces of image around (RAM-heavy)
    #iaa.LogContrast((0.5,1.0),True), #overlay color
    #iaa.MotionBlur(20,(0,288),1,0), #motion blur for realism
    #iaa.BlendAlpha((0.0, 1.0), 
    #iaa.MedianBlur(11), per_channel=True), #alpha-blending with median blur
    iaa.PerspectiveTransform(scale=(0.1, 0.1)),
    iaa.AdditiveGaussianNoise(scale=0.05*255, per_channel=True), #noise
    #iaa.CoarseDropout(p=0.1, size_percent=0.005), #blocks removed from image
    iaa.Affine(rotate=(-5,5)) #rotate #PROBLEM WITH BOUNDING BOXES MOSTLY CAUSED BY THIS
], random_order=True)

#---------------------------------------------------------------------

imagesToAugment = []
bbs_images = []

filepathOriginalImagesList = glob.glob(filepathOriginalFolder+"*.jpg")
filepathOriginalImagesList.sort()
filepathOriginalLabelsList = glob.glob(filepathOriginalFolder+"*.txt")
filepathOriginalLabelsList.sort()

# create bounding boxes on images
for i in range(len(filepathOriginalImagesList)):
    img = imageio.imread(filepathOriginalImagesList[i])
    imagesToAugment.append(img)

    f = open(filepathOriginalLabelsList[i], "r")
    contentsOfFile =f.read().split()
    for u in range(len(contentsOfFile)):
        contentsOfFile[u] = float(contentsOfFile[u])

    image_height, image_width = img.shape[:2]

    abs_x1,abs_x2,abs_y1,abs_y2 = convert_from_yolo(contentsOfFile[1],contentsOfFile[2],contentsOfFile[3],contentsOfFile[4],image_width,image_height)

    bb = BoundingBox(x1=abs_x1, x2=abs_x2, y1=abs_y1, y2=abs_y2, label = int(contentsOfFile[0]))
    bboi = BoundingBoxesOnImage([bb], shape=img.shape)
    bbs_images.append(bboi)

    f.close()

imagesToAugment= imagesToAugment*desiredMultiple
bbs_images = bbs_images*desiredMultiple

images_aug, bbs_aug = seq(images=imagesToAugment, bounding_boxes=bbs_images)

#check if bounding box is fully within augmented image
for i, image_aug in enumerate(images_aug):
    bb_info = bbs_aug[i].bounding_boxes[0]
    heightImage, widthImage = image_aug.shape[:2]
    widthImage = float(widthImage)
    heightImage = float(heightImage)
    if (not bb_info.is_fully_within_image(image_aug)):
        abs_x1,abs_y1,abs_x2,abs_y2 = saturate_bbs(bb_info.x1_int,bb_info.y1_int,bb_info.x2_int,bb_info.y2_int,widthImage,heightImage)
        bbs_aug[i].bounding_boxes[0] = BoundingBox(x1=abs_x1, x2=abs_x2, y1=abs_y1, y2=abs_y2, label = bb_info.label)


if (viewNBoundingBoxes > 0):
    for i in range(viewNBoundingBoxes):
        ia.imshow(bbs_aug[i].draw_on_image(images_aug[i], size=5))

#write to file, save image
for i, image_aug in enumerate(images_aug):
    f=open(filepath_txt % i,"w")

    bb_info = bbs_aug[i].bounding_boxes[0]
    heightImage, widthImage = image_aug.shape[:2]
    widthImage = float(widthImage)
    heightImage = float(heightImage)

    center_x,center_y,bbWidth,bbHeight = convert_to_yolo(bb_info.x1_int,bb_info.y1_int,bb_info.x2_int,bb_info.y2_int,widthImage,heightImage)
    
    f.write(str(bb_info.label) + ' ' + str(round(center_x,6)) + ' ' + str(round(center_y,6)) + ' ' + str(round(float(bbWidth),6)) + ' ' + str(round(float(bbHeight),6)))
    misc.imsave(filepath_img % i, image_aug)
    f.close()