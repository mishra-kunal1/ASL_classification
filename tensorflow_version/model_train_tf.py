def train_model(model,train_generator,val_generator,epochs):

  model.fit(train_generator, epochs=epochs, validation_data=val_generator)
  return model