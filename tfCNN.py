import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential


import pathlib


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=False)
data_dir = pathlib.Path(data_dir).with_suffix('')


image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 32
img_height = 180
img_width = 180
print(data_dir)

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    
)



#https://www.tensorflow.org/tutorials/images/classification

#https://paperswithcode.com/datasets?task=image-classification