from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def my_seq_model(n_classes):
  my_model = Sequential()
  my_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=(64,64,3)))
  my_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
  my_model.add(Dropout(0.5))
  my_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
  my_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
  my_model.add(Dropout(0.5))
  my_model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
  my_model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
  my_model.add(Flatten())
  my_model.add(Dropout(0.5))
  my_model.add(Dense(256, activation='relu'))
  my_model.add(Dense(n_classes, activation='softmax'))
  my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
  return my_model