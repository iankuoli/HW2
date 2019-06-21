from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np
import os
import cv2


# ----------------------------------------------------------------------------------------------------------------------
# Parameter Setting
#
batch_size = 16
num_classes = 5
epochs = 100
data_augmentation = False
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_HW2_trained_model.h5'


# ----------------------------------------------------------------------------------------------------------------------
# Load Data
#
file_names = []
for file in os.listdir("../data/jpg"):
    if file.endswith(".JPG"):
        file_names.append(file)

x_train = []
y_train = []
x_test = []
y_test = []

for file_name in file_names:

    # Data processing
    x = cv2.imread('../data/jpg/' + file_name, 0)
    x = cv2.resize(x, (256, 256), interpolation=cv2.INTER_CUBIC)

    # Data augmentation
    x_blur = cv2.GaussianBlur(x, (5, 5), 5)
    x_train.append(x)
    #x_train.append(cv2.flip(x, 1))
    #x_train.append(cv2.flip(x, 0))
    #x_train.append(cv2.flip(x, -1))
    x_train.append(x_blur)
    #x_train.append(cv2.flip(x_blur, 1))
    #x_train.append(cv2.flip(x_blur, 0))
    #x_train.append(cv2.flip(x_blur, -1))
    x_test.append(x)

    # Label processing
    y = int(file_name.split('-')[1].split('.')[0]) - 1
    y_train += [y] * 2
    y_test.append(y)

# The data, split between train and test sets:
x_train = np.expand_dims(np.array(x_train), axis=-1)
y_train = np.array(y_train)
x_test = np.expand_dims(np.array(x_test), axis=-1)
y_test = np.array(y_test)

index = np.arange(x_train.shape[0])
np.random.shuffle(index)
x_train = x_train[index, :, :, :]
y_train = y_train[index]

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# ----------------------------------------------------------------------------------------------------------------------
# Model Construction
#
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Initiate Adam optimizer
opt = keras.optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Train the model using Adam
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[early_stopping])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=len(x_train) // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4,
                        callbacks=[early_stopping])

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
