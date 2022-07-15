from __future__ import absolute_import, division, print_function, unicode_literals

#!pip install -q tensorflow==2.0.0-alpha0
import tensorflow as tf
import pathlib
import random
import os
import IPython.display as display






AUTOTUNE = tf.data.experimental.AUTOTUNE


data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
data_root = pathlib.Path(data_root_orig)
print(data_root)


for item in data_root.iterdir():
    print(item)


all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)
print(all_image_paths[:10])

attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)

def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])


for n in range(3):
  image_path = random.choice(all_image_paths)
  display.display(display.Image(image_path))
  print(caption_image(image_path))
  print()

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)

label_to_index = dict((name, index) for index, name in enumerate(label_names))
print(label_to_index)



all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
print("First 10 labels indices: ", all_image_labels[:10])















