
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preparing_dataset(path):
  data_dir = path
  target_size = (64, 64)  
  n_classes = 29
  val_frac = 0.2
  batch_size = 32
  data_augmentor = ImageDataGenerator(samplewise_center=True, 
                                    samplewise_std_normalization=True, 
                                    validation_split=val_frac)

  train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset="training")
  val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")
  return train_generator,val_generator

