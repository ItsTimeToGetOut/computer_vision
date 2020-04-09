## Validation not working
## Orignally by Aryaman Pande

import tensorflow as tf
import glob
import xmltodict
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from PIL import ImageDraw,Image
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

image_dim = 228
input_dim = image_dim
input_shape = (image_dim, image_dim, 3)
dropout_rate = 0.5
alpha = 0.2
num_classes = 1 ## initially was set to 3 
pred_vector_length = 4 + num_classes

images = []
for x in glob.glob('datasets/training_images/*.jpg'):
    image = (Image.open(x)).resize((image_dim,image_dim))
    image = np.asarray(image)
    images.append(image)

bboxes = []
classes = []
classes_raw=[]
xml_data = glob.glob( 'datasets/training_images/*.xml' )
for file in xml_data:
    x = xmltodict.parse(open(file,'rb'))
    bndbox = x[ 'annotation' ][ 'object' ][ 'bndbox' ]
    bounding_box = [None] * 4
    bounding_box[0] = int(bndbox['xmin']) / image_dim
    bounding_box[1] = int(bndbox['ymin']) / image_dim
    bounding_box[2] = int(bndbox['xmax']) / image_dim
    bounding_box[3] = int(bndbox['ymax']) / image_dim
    bboxes.append(bounding_box)
    classes_raw.append(x['annotation']['object']['name'])

boxes = np.array(bboxes)
encoder = LabelBinarizer() #like 1 hot encoder but it works on string classes too
classes_onehot = encoder.fit_transform(classes_raw)

Y = np.concatenate([boxes, classes_onehot], axis=1)
X = np.array(images)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)


def custom_loss( y_true , y_pred ):
    mse = tf.losses.mean_squared_error( y_true , y_pred )
    return mse


model_layers = [
	keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1, input_shape=input_shape),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1 ),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Flatten() ,

    keras.layers.Dense( 1240 ) ,
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Dense( 640 ) ,
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Dense( 480 ) ,
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Dense( 120 ) ,
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Dense( 62 ) ,
    keras.layers.LeakyReLU( alpha=alpha ) ,

    keras.layers.Dense( pred_vector_length ),
    keras.layers.LeakyReLU( alpha=alpha ) ,
]

model = keras.Sequential(model_layers)
model.compile(optimizer=keras.optimizers.Adam( lr=0.0001 ),loss=custom_loss,metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=( x_test , y_test ),epochs=100,batch_size=3)
model.save( 'model.h5')

boxes = model.predict(x_test)
for i in range(boxes.shape[0]):
    b = boxes[i, 0: 4] * input_dim
    img = x_test[i] * 255
    source_img = Image.fromarray(img.astype(np.uint8), 'RGB')
    draw = ImageDraw.Draw(source_img)
    draw.rectangle(b, outline="black")
    source_img.save('datasets/result_images/image_{}.png'.format(i + 1), 'png')

def class_accuracy( target_classes , pred_classes ):
    target_classes = np.argmax( target_classes , axis=1 )
    pred_classes = np.argmax( pred_classes , axis=1 )
    return ( target_classes == pred_classes ).mean()

target_boxes = y_test * input_dim
pred = model.predict( x_test )
pred_boxes = pred[ ... , 0 : 4 ] * input_dim
pred_classes = pred[ ... , 4 : ]
print( 'Class Accuracy is {} %'.format( class_accuracy( y_test[ ... , 4 : ] , pred_classes ) * 100 ))