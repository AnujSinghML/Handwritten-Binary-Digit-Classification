#Using Neural Networks for Handwritten Digit Recognition (Binary)
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

#load_data() function is defined in autils.py file
X, y = load_data()

##----------------uncomment the following part to check if data is loaded correctly.------------------
# print ('The first element of X is: ', X[0])
# print ('The first element of y is: ', y[0,0])
# print ('The last element of y is: ', y[-1,0])
##------------------------------------------------------------------------------------------------------


##uncomment the following part to visualize our dataset---------------------------------------------------
##Displaying the images (dataset):
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# m, n = X.shape
# fig, axes = plt.subplots(8,8, figsize=(8,8))
# fig.tight_layout(pad=1.0)
# for i,ax in enumerate(axes.flat):
#     # Select random indices
#     random_index = np.random.randint(m)
#     # Select rows corresponding to the random indices and
#     # reshape the image
#     X_random_reshaped = X[random_index].reshape((20,20)).T
#     # Display the image
#     ax.imshow(X_random_reshaped, cmap='gray')
#     # Display the label above the image
#     ax.set_title(y[random_index,0])
#     ax.set_axis_off()    
# plt.show()  
##-------------------------------------------------------------------------------------------------------

model = Sequential(
    [               
        tf.keras.Input(shape=(400,)), #specify input size
        
        tf.keras.layers.Dense(40, activation="sigmoid"),
        tf.keras.layers.Dense(25, activation="sigmoid"),
        tf.keras.layers.Dense(10, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")
      
    ], name = "my_model" 
) 

# [layer1, layer2, layer3, layer4] = model.layers
# W1,b1 = layer1.get_weights()
# W2,b2 = layer2.get_weights()
# W3,b3 = layer3.get_weights()
# W4,b4 = layer4.get_weights()
# print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
# print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
# print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")
# print(f"W4 shape = {W4.shape}, b4 shape = {b4.shape}")

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X,y,
    epochs=25
)

# #uncomment the following part to compare the predictions vs the labels for a random sample of 64 digits. This might take a moment to run--------------------------------------
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# m, n = X.shape
# fig, axes = plt.subplots(8,8, figsize=(8,8))
# fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]
# for i,ax in enumerate(axes.flat):
#     # Select random indices
#     random_index = np.random.randint(m)
#     # Select rows corresponding to the random indices and
#     # reshape the image
#     X_random_reshaped = X[random_index].reshape((20,20)).T
#     # Display the image
#     ax.imshow(X_random_reshaped, cmap='gray')
#     # Predict using the Neural Network------------------------------------------------------->
#     prediction = model.predict(X[random_index].reshape(1,400))
#     if prediction >= 0.5:
#         yhat = 1
#     else:
#         yhat = 0
#     # Display the label above the image
#     ax.set_title(f"{y[random_index,0]},{yhat}")
#     ax.set_axis_off()
# fig.suptitle("Label, yhat", fontsize=16)
# plt.show()
## --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
predictions = model.predict(X)
binary_predictions = (predictions >= 0.5).astype(int)
binary_predictions = binary_predictions.reshape(-1, 1)
correct_predictions = (binary_predictions == y).sum()

# Calculate the accuracy
accuracy = correct_predictions / y.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")


