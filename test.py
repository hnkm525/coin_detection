import os
import random
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import model_from_json

test_dir = 'images'
display_dir = 'images/100'
file_name = 'vgg16_coin_fine'
coins = ["1", "10", "100", "5", "50", "500"]
BATCH_SIZE = 64


model = tensorflow.keras.models.load_model('vgg16_coin_fine (1).h5')

model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#data generate
test_datagen=ImageDataGenerator(rescale=1.0/255)

test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

#evaluate model
score = model.evaluate_generator(test_generator)
print('\n test loss:',score[0])
print('\n test_acc:',score[1])

#predict model and display images
files=os.listdir(display_dir)
img=random.sample(files, 2)

plt.figure(figsize=(10,10))
for i in range(2):
    temp_img=load_img(os.path.join(display_dir,img[i]),target_size=(224,224))
    plt.subplot(5,5,i+1)
    plt.imshow(temp_img)
    #Images normalization
    temp_img_array=img_to_array(temp_img)
    temp_img_array=temp_img_array.astype('float32')/255.0
    temp_img_array=temp_img_array.reshape((1,224,224,3))
    #predict image
    img_pred=model.predict(temp_img_array)
    plt.title(coins[np.argmax(img_pred)] + "yen desu!")
    print(img_pred)
    #eliminate xticks,yticks
    plt.xticks([]),plt.yticks([])


plt.show()