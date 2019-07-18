from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from PIL import Image

import tensorflow as tf

print(tf.VERSION)

def get_input_paras():

  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--image_file", default="daisy.bmp", \
    help="image to recognize")
  parser.add_argument("-W", "--input_width", default=128, \
    help="image width the model accepts as input")
  parser.add_argument("-H", "--input_height", default=128, \
    help="image height the model accepts as input")
  parser.add_argument("-l", "--tflite_file", default="my_tflite_model.tflite", \
    help="Tflite model file used to recognize image")
  parser.add_argument("-v", "--verbose", default=0, \
    help="print more information if set to 1")
  args = parser.parse_args()

  return args

def prepare_image(args):

    img = Image.open(args.image_file)
    img = img.resize((args.input_width, args.input_height))

    # add N dim
    image = np.expand_dims(img, axis=0)
    image = (np.float32(image))
    image /= 255.

    return image

def recognize_image(args, img_tensor):

    # Create a tf-lite interpreater from the tf-lite model file.
    # We are invoking the Tf-lite model to recognize an image at RPI.

    f = tf.contrib.lite.Interpreter(args.tflite_file)

    f.allocate_tensors()
    i = f.get_input_details()[0]
    if int(args.verbose) == 1:
      print("=== input of tflite model: ", i)

    o = f.get_output_details()[0]
    if int(args.verbose) == 1:
      print("=== output of tflite model: ", o)

    f.set_tensor(i['index'], img_tensor)
    f.invoke()
    y = f.get_tensor(o['index'])
    if int(args.verbose) == 1:
      print("=== Results from recognizing the image with Tf-lite model: ", y)
    print("\n=== Label of the image that we recognize with Tf-lite model: {:d} \n".format(np.argmax(y)))

if __name__ == "__main__":

    # Parse and get parameters that you input on ternimal when you type:
    # python3 generate_my_tflite_model.py
    args = get_input_paras()

    img = prepare_image(args)

    recognize_image(args, img)
