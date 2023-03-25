import numpy as np
import matplotlib.pyplot as plt
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