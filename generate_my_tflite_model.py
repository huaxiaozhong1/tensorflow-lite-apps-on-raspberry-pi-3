import argparse
import tensorflow as tf
import numpy as np

print(tf.version.VERSION)

def get_input_paras():

  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--image_file", default="daisy.bmp", \
    help="image(bmp format) to recognize")
  parser.add_argument("-W", "--input_width", default=128, \
    help="image width the model accepts as input")
  parser.add_argument("-H", "--input_height", default=128, \
    help="image height the model accepts as input")
  parser.add_argument("-m", "--model_file", default="my_model.h5", \
    help="Tf keras model HDF5 file used to convert to tflite model")
  parser.add_argument("-l", "--tflite_file", default="my_tflite_model.tflite", \
    help="Tflite model file used to recognize image")
  parser.add_argument("-v", "--verbose", default=0, \
    help="print more information if set to 1")
  args = parser.parse_args()

  return args

def prepare_image(args):

    # preppare an image.
    img = tf.keras.preprocessing.image.load_img(args.image_file, target_size=(args.input_width, args.input_height))
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # Remember that the model was trained on inputs
    # that were preprocessed in the following way:
    img_tensor /= 255.

    return img_tensor

def verify_Tf_model(args, img_tensor):

    # predict the image with pure keras model.
    model = tf.keras.models.load_model(args.model_file)

    y = model.predict(img_tensor)
    if int(args.verbose) == 1:
      print("=== Results from recognizing the image with Tf model: ", y)
    print("\n=== Label of the image that we recognize with Tf model: {:d} \n".format(np.argmax(y)))

    return model

def convert_Tflite_model(args):

    # prepare a converter from the h5 keras modle file.
    converter = tf.lite.TFLiteConverter.from_keras_model_file
    tflite_model = converter(args.model_file).convert()
    # tflite_model = converter("model_full.h5").convert()

    # save the tf-lite keras model into a tf-lite model file.
    with open(args.tflite_file, 'wb') as f:
        f.write(tflite_model)

def verify_Tflite_model(args, img_tensor):

    # Create a tf-lite interpreater from the tf-lite model file.
    # Here we are invoking the Tf-lite model to recognize a image at host.
    # It simulates the action that will occur at RPI later.
    f = tf.lite.Interpreter(args.tflite_file)

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

    img_tensor = prepare_image(args)

    Tf_model = verify_Tf_model(args, img_tensor)

    convert_Tflite_model(args)

    verify_Tflite_model(args, img_tensor)
