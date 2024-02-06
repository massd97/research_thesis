#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
tf.__version__


# In[ ]:


file_path_t = r"D:\Sorghum\Series21\17126139\001\PGR_2022-07-08-195437.bmp"
file_path_f = r"D:\Sorghum\Series21\20202603\001\PGR_2022-07-08-195437.bmp"
# Load your images (replace these with your image paths)
image1 = tf.image.decode_bmp(tf.io.read_file(file_path_t), channels=4)
image2 = tf.image.decode_bmp(tf.io.read_file(file_path_f), channels=4)
image1 = image1[..., :3]
image2 = image2[..., :3]
height = width = 256
# Resize images to the same shape (if needed)
image1 = tf.image.resize(image1, [height, width])
image2 = tf.image.resize(image2, [height, width])

# Stack images horizontally
collage = tf.concat([image1, image2], axis=1)
collage = tf.cast(collage, tf.uint8)
# Save or display the collage
tf.io.write_file("D:\Sorghum\Series21\stacked\stacked_1.jpeg", tf.image.encode_jpeg(collage))


# def stack_img(i, saving_name):
#     output_folder = r"D:\Sorghum\Series21\stacked"
#     saving_img_path = os.path.join(output_folder, saving_name)
#     image1 = tf.image.decode_bmp(tf.io.read_file(image1), channels=4)
#     image2 = tf.image.decode_bmp(tf.io.read_file(image2), channels=4)
#     # Change channel number
#     image1 = image1[..., :3]
#     image2 = image2[..., :3]
#     height = width = 256
#     # Resize images to the same shape (if needed)
#     image1 = tf.image.resize(image1, [height, width])
#     image2 = tf.image.resize(image2, [height, width])
#     
#     # Stack images horizontally
#     collage = tf.concat([image1, image2], axis=1)
#     collage = tf.cast(collage, tf.uint8)
#     # Save or display the collage
#     tf.io.write_file(saving_img_path, tf.image.encode_jpeg(collage))
#     

# In[25]:


def stack_images(image_paths, output_folder):
    images = []
    for image_path in image_paths:
        image = tf.image.decode_bmp(tf.io.read_file(image_path), channels=4)[..., :3]
        image = tf.image.resize(image, [256, 256])
        images.append(image)

    # Stack images horizontally
    collage = tf.concat(images, axis=1)
    collage = tf.cast(collage, tf.uint8)

    # Generate a saving name based on the first image's filename
    saving_name = os.path.basename(image_paths[0])
    saving_path = os.path.join(output_folder, saving_name)

    # Save the collage
    tf.io.write_file(saving_path, tf.image.encode_jpeg(collage))


# In[20]:


img_1 = r"D:\Sorghum\Series21\17126139\001\PGR_2022-07-08-195437.bmp"
img_2 = r"D:\Sorghum\Series21\20202603\001\PGR_2022-07-08-195437.bmp"
sav_n = "try.bmp"
stack_img(img_1, img_2, sav_n)


# directory1 = r"D:\Sorghum\Series21\17126139"
# directory2 = r"D:\Sorghum\Series21\20202603"
# for (root1, dirs1, files1), (root2, dirs2, files2) in zip(os.walk(directory1), os.walk(directory2)):
#     for filename1, filename2 in zip(files1, files2):
#         if filename1.endswith(".bmp"):
#             saving_name = filename1
#             img_path_1 = os.path.join(root1, filename1)
#             img_path_2 = os.path.join(root2, filename2)
#             stack_img(img_path_1, img_path_2, saving_name)
#             #print(img_path_1, img_path_2)
#             #print(saving_name)

# In[26]:


directory1 = r"D:\Sorghum\Series21\17126139"
directory2 = r"D:\Sorghum\Series21\20202603"
output_folder = r"D:\Sorghum\Series21\stacked"

for (root1, _, files1), (root2, _, files2) in zip(os.walk(directory1), os.walk(directory2)):
    for filename1, filename2 in zip(files1, files2):
        if filename1.endswith(".bmp"):
            img_path_1 = os.path.join(root1, filename1)
            img_path_2 = os.path.join(root2, filename2)
            stack_images([img_path_1, img_path_2], output_folder)


# In[ ]:




