"""Package Installation"""
# !pip3 install matplotlib
# !pip3 install numpy
# !pip3 install pandas
# !pip3 install keras==2.3.1
# !pip3 install tensorflow==2.1.0
# !pip3 install scikit-learn==0.24.0

"""Import the Packages"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.utils.generic_utils import get_custom_objects
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

"""Import the Dataset"""
raw_data = pd.read_csv('data.csv')
raw_data.head()

"""Change the Y Values"""
# seizure = 1
# no_seizure = 0

# raw_data['y'].replace(1, seizure, inplace=True)
# raw_data['y'].replace(2, no_seizure, inplace=True)
# raw_data['y'].replace(3, no_seizure, inplace=True)
# raw_data['y'].replace(4, no_seizure, inplace=True)
# raw_data['y'].replace(5, no_seizure, inplace=True)

"""Split Dataset into Training Set and Test Set"""
# Percentage to split by for training
perc = 90
# Set for training the model
data = raw_data.head(int(len(raw_data) * (perc / 100)))
# Set for testing the model later to get the real accuracy
test_set = raw_data.tail(int(len(raw_data) * ((100 - perc) / 100)))

"""Format the Data
    1. Split the data into Train and Test sets
    2. Get the data into the right shapes for training"""
x_values = data.values[:, 1:-1]
y_values = np.array(data['y'])
y_values = np_utils.to_categorical(y_values)
x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=1)
x_train = x_train.reshape(-1, 178, 1)
x_test = x_test.reshape(-1, 178, 1)
print(
    "X Train: {}\nX Test: {}\nY Train: {}\nY Test {}".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

"""Visualize the Dataset"""
# Set the size of the chart
plt.figure(figsize=(12, 8))
# Plot data labeled 1 - Seizure
plt.plot(x_values[1, :], label="1 - Seizure")
# Plot data labeled 2 - No Seizure
plt.plot(x_values[7, :], label="2 - No Seizure")
# Plot data labeled 3 - No Seizure
plt.plot(x_values[12, :], label="3 - No Seizure")
# Plot data labeled 4 - No Seizure
plt.plot(x_values[0, :], label="4 - No Seizure")
# Plot data labeled 5 - No Seizure
plt.plot(x_values[2, :], label="5 - No Seizure")
# Create a legend and output the graph
plt.legend()
plt.show()

"""Model Training - Create a Custom Activation Function - Swish"""


def custom_activation(x, beta=2):
    """
    Define Swish Activation Function
    """
    return K.sigmoid(beta * x) * x


get_custom_objects().update({'custom_activation': Activation(custom_activation)})

"""Create a LSTM Model"""
# Create a Sequential LSTM model
model = Sequential()
model.add(LSTM(56, input_shape=(45, 1), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(56))
model.add(Dropout(0.3))
model.add(Dense(20))
# model.add(Activation(custom_activation, name='Swish'))
model.add(Activation('tanh'))
model.add(Dense(5))
model.add(Activation('softmax'))

# Output the model summary
model.summary()

"""Compile the LSTM Model"""

# Define the variables for training
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

"""Train the LSTM Model"""
# Train the model
hist = model.fit(
    x=np.asarray((x_train[:, ::4] - x_train.mean()) / x_train.std()).astype(np.float32),
    y=np.asarray(y_train[:, 1:]).astype(np.float32),
    validation_data=(
        (x_test[:, ::4] - x_test.mean()) / x_test.std(),
        y_test[:, 1:]
    ),
    epochs=100,
    batch_size=15,
    shuffle=True
)

"""Save the Model"""
model_name = "Seizure_Detector.h5"
model.save(model_name)

"""Visualize Model's Accuracy Metrics"""
# Plot Training Loss and Accuracy of the Model
plt.figure(0)
plt.plot(hist.history['loss'], 'green')
plt.plot(hist.history['accuracy'], 'red')
plt.show()

# Plot Validation Loss and Accuracy of the Model
plt.figure(0)
plt.plot(hist.history['val_loss'], 'blue')
plt.plot(hist.history['val_accuracy'], 'black')
plt.show()

"""Load the Model"""
model = load_model(model_name)

"""Split the Test Set"""
x_test_values = test_set.values[:, 1:-1]
y_test_values = np.array(test_set['y'])
y_test_values = np_utils.to_categorical(y_test_values)
x_test_values = x_test_values.reshape(-1, 178, 1)
print("x_test_values Shape: {}\ny_test_values Shape: {}".format(x_test_values.shape, y_test_values.shape))

"""Predict"""
predictions = model.predict((x_test_values[:, ::4] - x_test_values.mean()) / x_test_values.std())

"""Format the Data
    1. Get the data into the right input shapes for the predictions"""
y_pred = np.zeros((y_test_values.shape[0]))
y_truth = np.ones((y_test_values.shape[0]))

for i in range(y_test_values.shape[0]):
    y_pred[i] = np.argmax(predictions[i]) + 1
    y_truth[i] = np.argmax(y_test_values[i])

for i in range(y_test_values.shape[0]):
    if y_truth[i] != 1:
        y_truth[i] = 0
    if y_pred[i] != 1:
        y_pred[i] = 0

"""Calculate the Accuracy"""
print("Accuracy of the Model: " + str(accuracy_score(y_truth, y_pred)))
