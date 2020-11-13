# import numpy as np
# import pandas as pd
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import random
# import os
#
# from skimage.io import imread, imsave
# from skimage.transform import resize
# from skimage import color
#
# from pathlib import Path
#
# train_dir = "../dataset/train/"
# test_dir = "../dataset/test1/"
# print(os.listdir("../dataset"))
#
#
# import cv2
#
# def resize_images(input_dir, output_dir, size=(256,256)):
#     filenames = os.listdir(input_dir)
#     for filename in filenames:
#         if 'cat' in filename:
#             file_directory = output_dir + 'cat/'
#         else:
#             file_directory = output_dir + 'dog/'
#         image_data = cv2.imread(input_dir + filename).astype(np.uint8)
#         # resizing in float should it be converted to integer
#         resized_image = cv2.resize(image_data, size)
#         cv2.imwrite(file_directory+filename, resized_image)
#
#
# resize_images(train_dir, "../dataset/cats-vs-dogs/train/")


prg = "The concept behind gamification is not new, but certainly the advent of the word has been difficult. " \
      "The term “gamification” was “coined in 2002 by British consultant Nick Pelling, as a “deliberately ugly " \
      "word” to describe “apply gamelike accelerated user interface design to make electronic transactions both " \
      "enjoyable and fast” . This element of gamification can be considered from two different points of view. On " \
      "the one hand, we have the non-game context, which refers to the many fields where gamification can be" \
      " applied. On the other hand, the context refers also to the gaming environment where the player is immersed" \
      " and can fulfil game requirements. As we are going to see in the next chapters, game elements, design and" \
      " context represent the three main elements characterizing all the gamified experiences."

prg_lower = prg.lower()
word_list = prg_lower.split()
new_list = []

special_charac = ['“', '”', ',', '.', '-']

for words in word_list:
    for char in special_charac:
        words = words.strip(char)
        if char == '-':
            new_list.append(words)

# print(new_list)

repeat = []
count = 0
for words in new_list:
    count = new_list.count(words)
    repeat.append(count)

# print(repeat)


final_list = [['word', 'count']]
z = 0
while z in range(0, len(repeat)):
    prep_list = []
    i = z
    while i == z:
        prep_list.append(new_list[i])
        prep_list.append(repeat[i])
        i = 0.5
    final_list.append(prep_list)
    z += 1
print(final_list)




