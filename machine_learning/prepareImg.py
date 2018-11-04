#数据处理，切分训练集、测试集
import os
import numpy as np
import shutil
from imgaug import augmenters as iaa
import cv2
from keras.preprocessing import image

#%config ZMQInteractiveShell.ast_node_interactivity='none'
np.random.seed(2018)

project_root_dir = os.getcwd()

# root_train = os.path.join(project_root_dir, 'data_simple/train_split')
# root_train
# root_val = os.path.join(project_root_dir, 'data_simple/val_split')
# root_val
# root_total = os.path.join(project_root_dir, 'data_simple/train')
# root_total


root_train = os.path.join(project_root_dir, 'data/train_split')
root_train
root_val = os.path.join(project_root_dir, 'data/val_split')
root_val
root_total = os.path.join(project_root_dir, 'data/train')
root_total


allTargetTypes = ['rock', 'paper', 'scissors', 'room']

nbr_train_samples = 0
nbr_val_samples = 0

# 训练集，测试集比例
split_proportion = 0.7

# 图像上采样系数，upsampling
aug_mutil_time = 0

img_width = 128
img_height = 128



#增强规则
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

#     iaa.Fliplr(0.5), # horizontally flip 50% of the images

seq = iaa.Sequential([
#     iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
#     iaa.Flipud(0.2), # vertically flip 20% of all images
#     iaa.GaussianBlur(sigma=(0, 0.5)), # blur images with a sigma of 0 to 3.0
#      iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-180, 180), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees 去掉一个角
         )),

])


for targetType in allTargetTypes:
    print(targetType)
    if targetType in os.listdir(root_train):
        shutil.rmtree(os.path.join(root_train, targetType))
    
    if targetType not in os.listdir(root_train):
        os.mkdir(os.path.join(root_train, targetType))

    total_images = os.listdir(os.path.join(root_total, targetType))
    jpgFiles = []
    for names in total_images:
        if names.endswith(".jpg"):
            jpgFiles.append(names)
    total_images = jpgFiles

    nbr_train = int(len(total_images) * split_proportion)
    np.random.shuffle(total_images)
    train_images = total_images[:nbr_train]
    print("train_images", len(train_images))
    
    val_images = total_images[nbr_train:]
    print("val_images", len(val_images))

    for img in train_images:
        #print(type(img))
        nbr_train_samples=(nbr_train_samples+1)
        source = os.path.join(root_total, targetType, img)
        target = os.path.join(root_train, targetType, img)
        shutil.copy(source, target)
        for aug_batch_idx in range(aug_mutil_time):
            enhanceTargetPath = os.path.join(root_train, targetType, str(img).replace(".jpg","") +"_enhance_" + str(aug_batch_idx) + ".jpg")
            enhanceImg = image.load_img(source,target_size=(img_width, img_height))
            x = image.img_to_array(enhanceImg)
            x = np.expand_dims(x, axis=0)
            images_aug = seq.augment_images(x)  # done by the library
            cv2.imwrite(enhanceTargetPath ,images_aug[0])
    if targetType in os.listdir(root_val):
        shutil.rmtree(os.path.join(root_val, targetType))
    if targetType not in os.listdir(root_val):
        os.mkdir(os.path.join(root_val, targetType))
    for img in val_images:
        source = os.path.join(root_total, targetType, img)
        target = os.path.join(root_val, targetType, img)
        shutil.copy(source, target)
        nbr_val_samples += 1

print('Finish splitting train and val images!')
print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))

