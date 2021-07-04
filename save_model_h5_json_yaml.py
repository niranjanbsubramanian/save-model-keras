import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.datasets import fashion_mnist
from keras.utils import np_utils
 
batch_size = 128
num_classes = 10
epochs = 50
 
# input image dimensions
img_rows, img_cols = 28, 28
 
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
 
 
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
 
# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
 
#Building our CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
 
#compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.2,
          )
 
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# -------------------------- SAVE AND LOAD MODEL IN HDF5 FORMAT -------------------------- #
model.save('Fashion_Mnist_CNN.h5')

#loading the saved model
from keras.models import load_model
restored_model = load_model('/content/Fashion_Mnist_CNN.h5')

#make prediction
import numpy as np
index = np.random.randint(0, len(x_test)) # randomly choose a sample from the test set
sample = np.array([x_test[index]])
prediction = restored_model.predict(sample)
print('Sample Number:',index,'\n','\nPredicted Class:',np.argmax([prediction[0]]), 'Actual Class:',np.argmax(y_test[index]))

# -------------------------- SAVE AND LOAD MODEL IN JSON FORMAT -------------------------- #
# save the architecture
json_model = model.to_json()
with open('model.json','w') as json_file:
  json_file.write(json_model)

#saving model weights
model.save_weights('weights.h5')

#loading the JSON model
from keras.models import model_from_json

with open('/content/model.json', 'r') as json_file:
  json_string = json_file.read()
json_file.close()
json_model = model_from_json(json_string)

#loading the weights and compiling the model
json_model.load_weights('/content/weights.h5')

json_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#making the prediction
index = np.random.randint(0, len(x_test)) # Get a random index
sample = np.array([x_test[index]]) # select a random sample
prediction = json_model.predict(sample) # predict the sample
print('Sample Number:',index,'\n','\nPredicted Class:',np.argmax([prediction[0]]), 'Actual Class:',np.argmax(y_test[index]))

# -------------------------- SAVE AND LOAD MODEL IN YAML FORMAT -------------------------- #

# save architecture
yaml_model = model.to_yaml()
with open('model.yaml','w') as yaml_file:
  yaml_file.write(yaml_model)
#saving model weights
model.save_weights('weights.h5')

# model reconstruction from yaml
from keras.models import model_from_yaml
with open('/content/model.yaml','r') as yaml_file:
  yaml_string = yaml_file.read()
yaml_model = model_from_yaml(yaml_string)

#Load weights and make prediction
yaml_model.load_weights('/content/weights.h5')

yaml_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

index = np.random.randint(0, len(x_test)-1)
sample = np.array([x_test[index]])
prediction = yaml_model.predict(sample)
print('Sample Number:',index,'\n','\nPredicted Class:',np.argmax([prediction[0]]), 'Actual Class:',np.argmax(y_test[index]))
