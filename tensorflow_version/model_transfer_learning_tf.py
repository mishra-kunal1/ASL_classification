from tensorflow.keras import layers 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def define_model(classes):
  pre_trained_model = ResNet50(input_shape=(64,64,3), include_top=False, weights="imagenet")
  for layer in pre_trained_model.layers:
      layer.trainable = False
  last_layer=pre_trained_model.layers[-1]
  output_layer=last_layer.output
  x = layers.Flatten()(output_layer)
  #x=layers.Dense(128,activation='relu')(x)
  x=layers.Dense(classes,activation='softmax')(x)

  resnetmodel = Model(pre_trained_model.input, x)

  resnetmodel.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])
  #print(resnetmodel.summary())
  return resnetmodel