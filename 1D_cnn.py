import tensorflow as tf 
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from recording import Recording
import os
import numpy as np
import datetime
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

valid_labels= ["N", "V", "F"]


dest = '/home/faren/Documents/HB/Beats/'

def fix_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    # config.gpu_options.per_process_gpu_memory_fraction = 1.0
    session = tf.compat.v1.InteractiveSession(config=config)
fix_gpu()

data_list = []
label_list = []

for record_name in os.listdir(dest):

    record = Recording(record_name)
    record.load_beats(dest + record.name)
    beats = record.beats
    for beat_num in range(0, len(beats)):
        label = beats[beat_num]['type']
        data = beats[beat_num]['signal']
        data_list.append(data.reshape((300, 1)))
        label_list.append(valid_labels.index(label))

X = np.array(data_list)
y = np.array(label_list)
y = to_categorical(y, num_classes=5)

shuffle_index = np.random.permutation(len(X))
X = X[shuffle_index]
y = y[shuffle_index]

split_index = int(0.9 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# Define the model architecture
model = Sequential()
# batch normalization 
model.add(BatchNormalization(input_shape=(300, 1)))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(300, 1)))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
# high 
model.add(Dropout(rate=0.5))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(5, activation='softmax'))

# Define the Adam optimizer with specified parameters
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# adam = Adam(learning_rate=1e-3)

# Compile the model with the Adam optimizer and specified batch size
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=X_train, 
          y=y_train, 
          epochs=20, # change to 20
          batch_size=256,
          validation_data=(X_test, y_test), 
          callbacks=[tensorboard_callback])

test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=256)


y_pred_train = model.predict(X_train)
y_pred_train = np.argmax(y_pred_train, axis=1)

# Calculate the confusion matrix
cm_train = tf.math.confusion_matrix(np.argmax(y_train, axis=1), y_pred_train)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_train, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('CM for Train Data')
plt.show()

print('Test accuracy:', test_acc)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Calculate the confusion matrix
cm = tf.math.confusion_matrix(np.argmax(y_test, axis=1), y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('CM for Test Data')
plt.show()