from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Softmax
from keras.optimizers import Adam

# Define the model architecture
model = Sequential()
model.add(Conv1D(filters = 64, kernel_size = 5, activation = 'relu', input_shape = (300, 1)))
model.add(Conv1D(filters = 64, kernel_size = 5, activation = 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(rate = 0.5))
model.add(Conv1D(filters = 128, kernel_size = 3, activation = 'relu'))
model.add(Conv1D(filters = 128, kernel_size = 3, activation = 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(rate = 0.5))
model.add(Flatten())
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dropout(rate = 0.5))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(rate = 0.5))
model.add(Softmax(units = 5))

# Define the Adam optimizer with specified parameters
adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)

# Compile the model with the Adam optimizer and specified batch size
model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'], batch_size = 256)
