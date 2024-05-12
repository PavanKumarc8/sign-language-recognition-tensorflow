# Import Libraries
import string 
import pandas as pd 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Importing Dataset
def load_data(path): 
    df = pd.read_csv(path) 
    y = np.array([label if label < 9
                  else label-1 for label in df['label']]) 
    df = df.drop('label', axis=1) 
    x = np.array([df.iloc[i].to_numpy().reshape((28, 28)) 
                  for i in range(len(df))]).astype(float) 
    x = np.expand_dims(x, axis=3) 
    y = pd.get_dummies(y).values 
  
    return x, y 
  
X_train, Y_train = load_data('/content/sign_mnist_train.csv') 
X_test, Y_test = load_data('/content/sign_mnist_test.csv') 

# Model Development
model = tf.keras.models.Sequential([ 
    tf.keras.layers.Conv2D(filters=32, 
                           kernel_size=(3, 3), 
                           activation='relu', 
                           input_shape=(28, 28, 1)), 
    tf.keras.layers.MaxPooling2D(2, 2), 
  
    tf.keras.layers.Conv2D(filters=64, 
                           kernel_size=(3, 3), 
                           activation='relu'), 
    tf.keras.layers.MaxPooling2D(2, 2), 
  
    tf.keras.layers.Flatten(), 
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dense(256, activation='relu'), 
    tf.keras.layers.Dropout(0.3), 
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dense(24, activation='softmax') 
]) 

# Compile the Model
model.compile( 
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy'] 
) 

# Model Training
history = model.fit(X_train, 
                    Y_train, 
                    validation_data=(X_test, Y_test), 
                    epochs=5, 
                    verbose=1) 

# Model Evaluation
model.evaluate(X_test, Y_test)
