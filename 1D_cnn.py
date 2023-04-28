from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Softmax
from keras.optimizers import Adam
from keras.utils import to_categorical
from recording import Recording
import os
import numpy as np

dest = '/home/faren/Documents/HB/Beats/'

data_list = []
label_list = []

for record_name in os.listdir(dest):
    record = Recording(record_name)
    record.load_beats(dest + record.name)
    beats = record.beats
    for beat_num in range(0, len(beats)):
        label = beats[beat_num]['type']
        data = beats[beat_num]['signal']
        print('ok')
        data_list.append(data.reshape((300, 1)))
        label_list.append(label)

X = np.array(data_list)
y = np.array(label_list)

shuffle_index = np.random.permutation(len(X))
X = X[shuffle_index]
y = y[shuffle_index]

split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(X_train[0, 10])
print(X_train.shape)

# # Define the model architecture
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(300, 1)))
# model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(rate=0.5))
# model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
# model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(rate=0.5))
# model.add(Flatten())
# model.add(Dense(units=256, activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(Dense(units=128, activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(Softmax(units=5))

# # Define the Adam optimizer with specified parameters
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# # Compile the model with the Adam optimizer and specified batch size
# model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'], batch_size=256)

# # Train the model on the dataset
# model.fit(X_train, y_train, epochs=50, batch_size=256)

# test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=256)
# print('Test accuracy:', test_acc)