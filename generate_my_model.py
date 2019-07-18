# The program is inspired by the book "Deep Learning with Python"
# (https://books.google.com/books?id=Yo3CAQAACAAJ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Create, train and save a tensorflow model"""

import argparse
import numpy as np
import tensorflow as tf

print(tf.version.VERSION)

def get_input_paras():

  parser = argparse.ArgumentParser()
  parser.add_argument("-W", "--image_width", default=128, \
    help="image width as input to model")
  parser.add_argument("-H", "--image_height", default=128, \
    help="image height as input to model")
  parser.add_argument("-C", "--image_channels", default=3, \
    help="image channels as input to model")
  parser.add_argument("-r", "--rotation_range", default=5, \
    help="image rotation range for data augmentation")
  parser.add_argument("-ws", "--width_shift_range", default=0.1, \
    help="image width shift range for data augmentation")
  parser.add_argument("-hs", "--height_shift_range", default=0.1, \
    help="image height shift range for data augmentation")
  parser.add_argument("-s", "--shear_range", default=0.1, \
    help="image shear range for data augmentation")
  parser.add_argument("-z", "--zoom_range", default=0.1, \
    help="image zoom range for data augmentation")
  parser.add_argument("-t", "--train_dir", \
    default="./flower_photos/train", \
    help="directory to get training data")
  parser.add_argument("-v", "--val_dir", default="./flower_photos/validation", \
    help="directory to get validation data")
  parser.add_argument("-b", "--batch_size", default=64, \
    help="batch size for data generator")
  parser.add_argument("-a", "--train_acc", default=0.96, \
    help="accuracy for training to reach")
  parser.add_argument("-va", "--val_acc", default=0.80, \
    help="accuracy for validation to reach")
  parser.add_argument("-e", "--epochs", default=5000, \
    help="iteration number the training runs over all the data")
  parser.add_argument("-m", "--model_file", default="my_model.h5", \
    help="file we finally save the mode to")
  args = parser.parse_args()

  return args

def create_model(args):

  model = tf.keras.models.Sequential()

  model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
      input_shape=(int(args.image_width), int(args.image_height), int(args.image_channels))))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))

  model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))

  model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
  model.add(tf.keras.layers.Dropout(0.5))

  model.add(tf.keras.layers.Dense(10, activation='softmax'))

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.RMSprop(lr=0.00001),
                metrics=['accuracy'])

  print(model.summary())

  return model

def collect_data(args):
  # Since we have a small data set, data augmentation is imported.
  # Although the api is specific to image processing, the principle
  # can help other kind of applications.
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255,
      rotation_range=int(args.rotation_range),
      width_shift_range=int(args.width_shift_range),
      height_shift_range=int(args.height_shift_range),
      shear_range=int(args.shear_range),
      zoom_range=int(args.zoom_range))

  test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

  # The generator control data to pump out 1 btach once.
  train_generator = train_datagen.flow_from_directory(
    args.train_dir,
    target_size=(int(args.image_width), int(args.image_height)),
    batch_size=int(args.batch_size),
    class_mode='sparse')

  validation_generator = test_datagen.flow_from_directory(
    args.val_dir,
    target_size=(int(args.image_width), int(args.image_height)),
    batch_size=int(args.batch_size),
    class_mode='sparse')

  return train_generator, validation_generator

if __name__ == "__main__":

  # Parse and get parameters that you input on ternimal when you type:
  # python3 generate_my_model.py
  args = get_input_paras()

  # Create a convolutional neural network model. It is a 2D, 3 layers model
  # based at tf.keras APIs. But you could replace it with any of your own ones.
  model = create_model(args)

  # Collect your own data for training and validation. That is to create
  # train-set and test-set.
  train_generator, validation_generator = collect_data(args)

  # Epoch is an iteration that model is trained with all data.
  # As soon as an epoch is ended, the following callback will be called.
  # We use of the chance to print out the accuracy on train-set and
  # validation-set, and etc.
  class MyCustomCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
          print(' === epoch {}, val_loss{:7.4f}, val_acc{:7.4f}.'.format(epoch, logs['val_loss'], logs['val_acc']))
          # Set criteria to stop traing.
          if ((logs['val_acc'] > int(args.val_acc)) or (logs['acc'] > int(args.train_acc))):
            self.stopped_epoch = epoch
            self.model.stop_training = True

  model.fit_generator(
    train_generator,
    epochs=int(args.epochs),
    validation_data=validation_generator,
    callbacks=[MyCustomCallback()])

  # As soon as the training stops, save the modle into a HDF5 file.
  tf.keras.models.save_model(model, args.model_file)
