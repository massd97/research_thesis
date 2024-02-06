#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display
import tensorflow as tf

from sklearn.model_selection import train_test_split

tf.config.list_physical_devices()


# In[3]:


def load_image(img_file):
    crop_size = (0,256,0,256)
    crop_size_t = (0,256,0,256)
    img = tf.io.read_file(img_file)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize(img, [256, 512])
    #print(img.shape)

    width = tf.shape(img)[1]
    #print(width)
    width = width // 2
    #print(width)
    original_img = img[:, :width, :]
    transformed_img = img[:, width:, :]
    
    original_img = tf.image.crop_to_bounding_box(original_img,0, 0, 256, 200)
    transformed_img = tf.image.crop_to_bounding_box(transformed_img, 0, 25, 200, 175)
  
    original_img = tf.cast(original_img, tf.float32)
    transformed_img = tf.cast(transformed_img, tf.float32)

    return original_img, transformed_img


# crop_size = (0,0,0,0)
# crop_size_t = (0,0,0,0)
# original_img = tf.image.crop_to_bounding_box(img_file[0], crop_size[1], crop_size[0], crop_size[3] - crop_size[1], crop_size[2] - crop_size[0])
# transformed_img = tf.image.crop_to_bounding_box(img_file[1], crop_size_t[1], crop_size_t[0], crop_size_t[3] - crop_size_t[1], crop_size_t[2] - crop_size_t[0])
# 

# In[4]:


file_path = r"D:\Sorghum\Series21\stacked"
features = labels = []
whole = []
for (root, dirs, files) in os.walk(file_path):
        for f in files:
            path = os.path.join(root, f)
            #feature, label = load_image(path)
            #features.append(feature)
            #labels.append(label)
            whole.append(path)


# In[5]:


train, test = train_test_split(whole, test_size=.2, random_state=42, shuffle=True)


# In[6]:


len(train), len(test)


# In[7]:


directory = r"D:\Sorghum\Series21\stacked\*.bmp"
quantity_training = tf.data.Dataset.list_files(train)
quantity_training = len(list(quantity_training))
quantity_training


# In[8]:


buffer_size = quantity_training
batch_size = 1
img_width = 256
img_height = 256


# In[9]:


def resize(original_img, transformed_img, width, height):
  original_img = tf.image.resize(original_img, [width, height], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  transformed_img = tf.image.resize(transformed_img, [width, height], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return original_img, transformed_img
    
def normalize(original_img, transformed_img):
  original_img = (original_img / 127.5) - 1
  transformed_img = (transformed_img / 127.5) - 1
  return original_img, transformed_img

def random_crop(original_img, transformed_img):
  stacked_img = tf.stack([original_img, transformed_img], axis = 0)
  crop_img = tf.image.random_crop(stacked_img, size = [2, img_width, img_height, 3])
  return crop_img[0], crop_img[1]


# In[10]:


directory = r"D:\Sorghum\Series21\stacked\PGR_2022-07-11-085231.bmp"
original_img, transformed_img = load_image(directory)
original_img, transformed_img = resize(original_img, transformed_img, 256, 256)
plt.figure()
plt.imshow(original_img / 255.0)
plt.figure()
plt.imshow(transformed_img / 255.0);


# In[11]:


@tf.function()
def random_jitter(original_img, transformed_img):
  original_img, transformed_img = resize(original_img, transformed_img, 256, 256)
  original_img, transformed_img = random_crop(original_img, transformed_img)
  if tf.random.uniform(()) > 0.5:
    original_img = tf.image.flip_left_right(original_img)
    transformed_img = tf.image.flip_left_right(transformed_img)
  return original_img, transformed_img


# In[12]:


plt.figure(figsize = (10,6))
for i in range(6):
  j_original, j_transformed = random_jitter(original_img, transformed_img)
  plt.subplot(2,3, i + 1)
  plt.imshow(j_transformed / 255.0)
  plt.axis('off')
plt.show()


# In[13]:


def load_training_images(img_file):
  original_img, transformed_img = load_image(img_file)
  original_img, transformed_img = random_jitter(original_img, transformed_img)
  original_img, transformed_img = normalize(original_img, transformed_img)
  return original_img, transformed_img


# In[14]:


def load_testing_images(img_file):
  original_img, transformed_img = load_image(img_file)
  original_img, transformed_img = resize(original_img, transformed_img, img_width, img_height)
  original_img, transformed_img = normalize(original_img, transformed_img)
  return original_img, transformed_img


# In[15]:


training_dataset = tf.data.Dataset.list_files(train)
training_dataset = training_dataset.map(load_training_images, num_parallel_calls=tf.data.AUTOTUNE)
training_dataset = training_dataset.shuffle(buffer_size)
training_dataset = training_dataset.batch(batch_size)


# In[16]:


testing_dataset = tf.data.Dataset.list_files(test)
testing_dataset = testing_dataset.map(load_testing_images)
testing_dataset = testing_dataset.batch(batch_size)


# In[17]:


training_dataset


# In[18]:


def encode(filters, size, apply_batchnorm = True):
  initializer = tf.random_normal_initializer(0, 0.02)
  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2D(filters, size, strides = 2, padding = 'same',
                                    kernel_initializer=initializer, use_bias=False))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())
  return result


# In[19]:


def decode(filters, size, apply_dropout = False):
  initializer = tf.random_normal_initializer(0, 0.02)
  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides = 2, padding = 'same',
                                             kernel_initializer=initializer, use_bias=False))
  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


# In[20]:


def Generator():
  inputs = tf.keras.layers.Input(shape = [256, 256, 3])

  downsampling = [
      encode(64, 4, apply_batchnorm = False), # (batch_size, 128, 128, 64)
      encode(128, 4), # (batch_size, 64, 64, 128)
      encode(256, 4), # (batch_size, 32, 32, 256)
      encode(512, 4), # (batch_size, 16, 16, 512)
      encode(512, 4), # (batch_size, 8, 8, 512)
      encode(512, 4), # (batch_size, 4, 4, 512)
      encode(512, 4), # (batch_size, 2, 2, 512)
      encode(512, 4), # (batch_size, 1, 1, 512)
  ]

  upsampling = [
      decode(512, 4, apply_dropout=True), # (batch_size, 2, 2, 512)
      decode(512, 4, apply_dropout=True), # (batch_size, 4, 4, 512)
      decode(512, 4, apply_dropout=True), # (batch_size, 8, 8, 512)
      decode(512, 4), # (batch_size, 16, 16, 512)
      decode(256, 4), # (batch_size, 32, 32, 256)
      decode(128, 4), # (batch_size, 64, 64, 128)
      decode(64, 4), # (batch_size, 128, 128, 64)
  ]

  output_channels = 3
  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same',
                                         kernel_initializer=initializer, activation='tanh') # (batch_size, 256, 256, 3)

  x = inputs
  skips = []
  for down in downsampling:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  for up, skip in zip(upsampling, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs = inputs, outputs = x)


# In[21]:


generator = Generator()
generator.summary()


# In[22]:


g_output = generator(original_img[tf.newaxis, ...], training=False)
plt.imshow(g_output[0, ...])


# In[3]:


lr = 0.0002
beta1, beta2 = 0.5, 0.999
lambda_ = 100


# In[24]:


loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)


# In[51]:


def generator_loss(d_generated_output, g_output, target):
  gan_loss = loss(tf.ones_like(d_generated_output), d_generated_output)
  l1_loss = tf.reduce_mean(tf.abs(target - g_output)) # MAE
  g_loss_total = gan_loss + (lambda_ * l1_loss) # Generator_loss
  return g_loss_total, gan_loss, l1_loss


# In[26]:


def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  original = tf.keras.layers.Input(shape = [256,256,3], name='original_img')
  generated = tf.keras.layers.Input(shape=[256,256,3], name='generated_img')
  x = tf.keras.layers.concatenate([original, generated]) # (batch_size, 256, 256, channels * 2)

  down1 = encode(64, 4, False)(x) # (batch_size, 128, 128, 64)
  down2 = encode(128, 4)(down1) # (batch_size, 64, 64, 128)
  down3 = encode(256, 4)(down2) # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides = 1, kernel_initializer=initializer, use_bias=False)(zero_pad1) # (batch_size, 31, 31, 512)
  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (batch_size, 33, 33, 512)
  last = tf.keras.layers.Conv2D(1, 4, strides = 1, kernel_initializer=initializer)(zero_pad2) # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs = [original, generated], outputs = last)


# In[27]:


discriminator = Discriminator()
discriminator.summary()


# In[28]:


d_output = discriminator([original_img[tf.newaxis, ...], g_output], training=False)
plt.imshow(d_output[0, ..., -1])


# In[29]:


def discriminator_loss(d_real_output, d_generated_output):
  real_loss = loss(tf.ones_like(d_real_output), d_real_output)
  generated_loss = loss(tf.zeros_like(d_generated_output), d_generated_output)
  d_total_loss = real_loss + generated_loss
  return d_total_loss


# In[4]:


generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1 = beta1, beta_2 = beta2)
discriminator_optimizer = tf.keras.optimizers.Adam(lr, beta_1 = beta1, beta_2 = beta2)


# In[31]:


checkpoint_dir = r"./checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generator = generator,
                                 discriminator = discriminator)


# In[32]:


def generate_images(model, test_input, real, step = None):
    generated_img = model(test_input, training=True)
    
    plt.figure(figsize=(12,8))
    
    img_list = [test_input[0], real[0], generated_img[0]]
    title = ['Input image(Real_top)', 'Real_front', 'Generated_image']
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(img_list[i] * 0.5 + 0.5)
        plt.axis('off')
        #plt.savefig("not_trained")
    
    if step is not None:
        plt.savefig('results/result_pix2pix_step_{}.png'.format(step), bbox_inches='tight')
        
    plt.show()
    return img_list[1], img_list[2]


# In[33]:


for input_example, real_example in testing_dataset.take(2):
    generate_images(generator, input_example, real_example)


# In[34]:


path_log = 'new_logs2/'
metrics = tf.summary.create_file_writer(path_log + 'fit/' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))


# In[35]:


@tf.function
def training_step(input_img, real, step):
  with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
    g_output = generator(input_img, training = True)

    d_output_real = discriminator([input_img, real], training = True)
    d_output_generated = discriminator([input_img, g_output], training = True)

    g_total_loss, g_loss_gan, g_loss_l1 = generator_loss(d_output_generated, g_output, real)
    d_loss = discriminator_loss(d_output_real, d_output_generated)

  generator_gradients = g_tape.gradient(g_total_loss, generator.trainable_variables)
  discriminator_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

  with metrics.as_default():
    tf.summary.scalar('g_total_loss', g_total_loss, step=step//1000)
    tf.summary.scalar('g_loss_gan', g_loss_gan, step=step//1000)
    tf.summary.scalar('g_loss_l1', g_loss_l1, step=step//1000)
    tf.summary.scalar('d_loss', d_loss, step=step//1000)


# In[36]:


def train(training_dataset, testing_dataset, steps):
    start = time.time()
    
    for step, (input_img, real_img) in training_dataset.repeat().take(steps).enumerate():
        test_input, real_input = next(iter(testing_dataset.take(1)))
        
        if step % 1000 == 0:
            #display.clear_output(wait = True)
            if step != 0:
                print(f'Time taken to run 1000 steps: {time.time() - start:.2f} seconds\n')
            start = time.time()
            generate_images(generator, test_input, real_input, step)
            print(f'Step: {step//1000}K')
        training_step(input_img, real_img, step)
        if (step + 1) % 10 == 0:
            print('.', end = '', flush = True)
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            generator.save_weights('new_model_pix2pix.h5')


# In[37]:


train(training_dataset, testing_dataset, steps = 40000) # 40000


# In[38]:


model2 = Generator()


# In[ ]:


model2.load_weights('new_model_pix2pix.h5')

real = []
gen = []
for top, front in testing_dataset.take(150):
    real_s, gen_s = generate_images(model2, top, front)
    real.append(real_s)
    gen.append(gen_s)


# In[41]:


plt.imshow(real[6])


# In[42]:


plt.imshow(gen[6])


# In[43]:


import cv2
import torch, numpy as np


# In[ ]:


real_hsv = []
gen_hsv = []
for real_s, gen_s in zip(real, gen):
    np_image1 = real_s.numpy()
    np_image2 = gen_s.numpy()
    image1 = (np_image1 * 255).astype(np.uint8)
    image2 = (np_image2 * 255).astype(np.uint8)
    image1 = image1.reshape([256, 256, 3])
    image2 = image2.reshape([256, 256, 3])
    image1.shape, image2.shape
    
    hsv_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    real_hsv.append(hsv_image1)
    gen_hsv.append(hsv_image2)


lower_green = np.array([20, 37, 65])
upper_green = np.array([120, 255, 255])

#lower_green = np.array([35, 50, 50])
#upper_green = np.array([85, 255, 255])
# Create a mask for the green regions
real_green_percent = []
gen_green_percent = []
for real, gen in zip(real_hsv, gen_hsv):
    mask1 = cv2.inRange(real, lower_green, upper_green)
    mask2 = cv2.inRange(gen, lower_green, upper_green)
    
    # Calculate the percentage of green area
    green_area1 = np.sum(mask1 == 255)
    total_area1 = mask1.size
    green_percentage1 = (green_area1 / total_area1) * 100
    green_area2 = np.sum(mask2 == 255)
    total_area2 = mask2.size
    green_percentage2 = (green_area2 / total_area2) * 100
    real_green_percent.append(green_percentage1)
    gen_green_percent.append(green_percentage2)
    print(f"{green_percentage1}% {green_percentage2}%")


# In[ ]:





# In[52]:


real_green_percent = np.array(real_green_percent)
gen_green_percent = np.array(gen_green_percent)

coefficient = np.polyfit(real_green_percent, gen_green_percent, 1)
slope = coefficient[0]
intercept = coefficient[1]
plt.scatter(real_green_percent, gen_green_percent)
plt.plot(real_green_percent, slope * real_green_percent + intercept, color='red', label='回帰直線')
plt.xlabel("real images(%)")
plt.ylabel("generated images(%)")
plt.title("Comparison amount of green between Real and Generated images")
plt.show()
plt.savefig("comparison_green_amount")


# In[53]:


coefficient, slope


# In[60]:


#lower_green = np.array([20, 37, 65])
#upper_green = np.array([120, 255, 255])
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])
mask1 = cv2.inRange(real_hsv[5], lower_green, upper_green)
mask2 = cv2.inRange(gen_hsv[5], lower_green, upper_green)

# Calculate the percentage of green area
green_area1 = np.sum(mask1 == 255)
total_area1 = mask1.size
green_percentage1 = (green_area1 / total_area1) * 100
green_area2 = np.sum(mask2 == 255)
total_area2 = mask2.size
green_percentage2 = (green_area2 / total_area2) * 100


# In[58]:


plt.subplot(1, 2, 1)
plt.title("real")
plt.imshow(real_hsv[5])
plt.subplot(1, 2, 2)
plt.title("generated")
plt.imshow(gen_hsv[5])
plt.show()


# In[59]:


plt.subplot(1, 2, 1)
plt.title("real")
plt.imshow(mask1)
plt.subplot(1, 2, 2)
plt.title("generated")
plt.imshow(mask2)
plt.show()


# In[61]:


plt.subplot(1, 2, 1)
plt.title("real")
plt.imshow(mask1)
plt.subplot(1, 2, 2)
plt.title("generated")
plt.imshow(mask2)
plt.show()


# In[ ]:





# In[123]:


tensorboar

