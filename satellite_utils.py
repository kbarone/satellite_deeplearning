import numpy as np
import matplotlib.pyplot as plt
import random

import os
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
import random

def hex_to_rgb(class_type):
    """
    Convert hex color value to RGB
    Inputs 
        class_type: hex color (string)
    Returns
        rbg color array (1D numpy array) 
    """
    class_type = class_type.lstrip("#")
    class_type = np.array(tuple(int(class_type[i:i+2], 16) for i in (0,2,4))) 
    return class_type

def rgb_to_label(label, class_dict_rgb):
    '''
    Convert RGB label array into one hot encoded label array
    Inputs
        label: 3D RGB label array
    Returns
        label_segment: 2D one-hot-encoded label array
  '''
    label_segment = np.zeros(label.shape, dtype=np.uint8)
    for i, (key, val) in enumerate(class_dict_rgb.items()):
        label_segment[np.all(label == val, axis=-1)] = i
    label_segment = label_segment[:,:,0]
    return label_segment

def plot_random_image(dataset, labels):
    '''
    Plot a random image and it's corresponding label
    Inputs
        dataset: (array) Array of 3D images
        labels: (array) Array of 2D label masks
    Returns
        None, plots image and corresponding label
    '''
    random_image_id = random.randint(0, len(dataset))
    plt.figure(figsize=(14,8))
    plt.subplot(121)
    plt.imshow(dataset[random_image_id])
    plt.subplot(122)
    plt.imshow(labels[random_image_id])

def to_categorical(labels, num_classes):
    '''
    One-hot encodes a tensor
    Inputs
        y: (array) Array of 2D mask labels
        num_classes: (int) unique number of classes
    Returns
        Array of "num_classes"-dimensional mask labels
    '''
    return np.eye(num_classes, dtype='uint8')[labels]


#!pip install patchify
#!pip install segmentation_models_pytorch

minmaxscaler = MinMaxScaler()
#dataset_root_folder = '/content/drive/MyDrive/satellite'
dataset_root_folder = '/Users/katybarone/Documents/uchicago/projects/satellite_deeplearning/semantic_segmentation_dataset'
image_patch_size = 256

def run():
    '''
    Read in and process satellite images
    Inputs: None
    Returns
        image_dataset: (array) array of 3D images
        labels: (array) array of 2D mask labels
    '''
    image_dataset = []
    mask_dataset = []

    for typ in ['images', 'masks']:
        if typ == 'images':

            ext = 'jpg'
        else:
            ext = 'png'
        for tile_id in range(1,8):
            for image_id in range(1,20):
                image = cv2.imread(f'{dataset_root_folder}/Tile {tile_id}/{typ}/image_part_00{image_id}.{ext}', 1)
                if image is not None:
                    if typ == 'masks':
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    size_x = (image.shape[1]//image_patch_size)*image_patch_size
                    size_y = (image.shape[0]//image_patch_size)*image_patch_size
                    image = Image.fromarray(image)
                    image = image.crop((0,0,size_x,size_y))
                    image = np.array(image)
                    image_patches = patchify(image, (image_patch_size, image_patch_size, 3), 
                                                    step=image_patch_size)
                    for i in range(image_patches.shape[0]):
                        for j in range(image_patches.shape[1]):
                            if typ == 'images':
                                ind_patched_img = image_patches[i,j,:,:]
                                ind_patched_img = minmaxscaler.fit_transform(ind_patched_img.reshape(-1, 
                                                                            ind_patched_img.shape[-1])).reshape(
                                                                                ind_patched_img.shape)
                                ind_patched_img = np.squeeze(ind_patched_img)
                                image_dataset.append(ind_patched_img)
                            else:
                                ind_patched_mask = image_patches[i,j,:,:]
                                ind_patched_mask = np.squeeze(ind_patched_mask)
                                mask_dataset.append(ind_patched_mask)

    image_dataset = np.array(image_dataset)
    mask_dataset = np.array(mask_dataset)

    # plot a random image and its corresponding label
    plot_random_image(image_dataset, mask_dataset)

    # create dictionary of RGB values for each class
    class_dict = {'building':'#3C1098', 'land':'#8429F6', 
                'road': '#6EC1E4', 'vegetation':'#FEDD3A', 
                'water':'#E2A929', 'unlabeled':'#9B9B9B'}
    class_dict_rgb = {}
    for key, val in class_dict.items():
        class_dict_rgb[key] = class_dict_rgb.get(key, hex_to_rgb(val))

    labels = []
    for i in range(mask_dataset.shape[0]):
        label = rgb_to_label(mask_dataset[i], class_dict_rgb)
        labels.append(label)

    labels = np.expand_dims(labels, axis=3)

    plot_random_image(image_dataset, labels)

    return image_dataset, labels


