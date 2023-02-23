import model_custom_tf
import model_transfer_learning_tf
import model_train_tf
import preparing_dataset_tf

num_classes=29
path='/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/'
#path='../asl_alphabet_train/asl_alphabet_train/'
train_loader, val_loader = preparing_dataset_tf.preparing_dataset(path)
print('Data Loaded')

curr_model=model_transfer_learning_tf.define_model(num_classes)
print('Model Loaded')
print('Training Started')
epochs=10
model_train_tf.train_model(curr_model,train_loader,val_loader,epochs)
