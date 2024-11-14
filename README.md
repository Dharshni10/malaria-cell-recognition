# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
The objective is to develop a system that can automatically categorize images of red blood cells into two distinct groups: those that are infected with malaria (parasitized) and those that are healthy (uninfected). The presence of the Plasmodium parasite in infected cells differentiates them from healthy cells. This initiative aims to leverage convolutional neural networks (CNNs) to effectively identify these categories. Traditional methods of examining blood smears for malaria diagnosis are labor-intensive and susceptible to human error. By implementing deep learning techniques, the process can be automated, which will enhance the speed of diagnosis, alleviate the workload on healthcare professionals, and improve the accuracy of detection.
The dataset utilized for training and testing comprises 27,558 labeled cell images, with an equal distribution between parasitized and uninfected samples. This balanced dataset serves as a solid foundation for training the model. The automation of this classification task is expected to streamline the diagnostic workflow significantly, addressing the challenges posed by manual inspections.

## Neural Network Model

![image](https://github.com/user-attachments/assets/a46f99a6-cf77-4646-a517-62bd4eb002ab)

### Design Steps
1. **Import Libraries**:Import TensorFlow, data preprocessing tools, and visualization libraries.
2. **Configure GPU**:Set up TensorFlow for GPU acceleration to speed up training.
3. **Data Augmentation**:Create an image generator for rotating, shifting, rescaling, and flipping to enhance model generalization.
4. **Build CNN Model**:Design a convolutional neural network with convolutional layers, max-pooling, and fully connected layers; compile the model.
5. **Train Model**:Split the dataset into training and testing sets, then train the model using the training data.
6. **Evaluate Performance**:Assess the model using the testing data, generating a classification report and confusion matrix.

## PROGRAM

### Name: Dharshni V M

### Register Number: 212223240029

```
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
# to share the GPU resources for multiple sessions
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

%matplotlib inline
my_data_dir = 'dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
os.listdir(train_path)
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
os.listdir(train_path+'/parasitized')[0]
para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[0])plt.imshow(para_img)
plt.imshow(para_img)
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

sns.jointplot(x=dim1,y=dim2)
image_shape = (130,130,3)
help(ImageDataGenerator)

image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
##name:Dharshni V M
#regno:212223240029
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output and add Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid for binary classification
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
batch_size = 16
help(image_gen.flow_from_directory)

train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')
train_image_gen.batch_size
len(train_image_gen.classes)
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)
train_image_gen.class_indices
results = model.fit(train_image_gen,epochs=2,
                              validation_data=test_image_gen
                             )
model.save('cell_model.h5')
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
model.metrics_names
model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
test_image_gen.classes
predictions = pred_probabilities > 0.5
print(classification_report(test_image_gen.classes,predictions))
confusion_matrix(test_image_gen.classes,predictions)
import random
import tensorflow as tf
list_dir=["uninfected"]
dir_=(random.choice(list_dir))
para_img= imread(train_path+
                 '/'+dir_+'/'+
                 os.listdir(train_path+'/'+dir_)[random.randint(0,100)])
img  = tf.convert_to_tensor(np.asarray(para_img))
img = tf.image.resize(img,(130,130))

img=img.numpy()
pred=bool(model.predict(img.reshape(1,130,130,3))<0.5)

print("Dharshni V M \n 212223240029")
plt.title("Model prediction: "+("Parasitized" if pred  else "Uninfected")+"\nActual Value: "+str(dir_))
plt.axis("off")
plt.imshow(img)
plt.show()

```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2024-11-14 084032](https://github.com/user-attachments/assets/8e08985d-3152-4303-897f-c0b1425d12fc)

### Classification Report

![Screenshot 2024-11-14 084146](https://github.com/user-attachments/assets/6d3c269e-e462-42b1-8f82-26a1c54e1c57)

### Confusion Matrix

![Screenshot 2024-11-14 084221](https://github.com/user-attachments/assets/a0e0a934-e4d5-45e5-9089-c60b54f155eb)

### New Sample Data Prediction

![Screenshot 2024-11-14 084250](https://github.com/user-attachments/assets/b6fd1b00-4895-441d-b4d1-8a6ee7ebd340)

## RESULT
Thus a deep neural network for Malaria infected cell recognition and to analyze the performance is created using tensorflow.

