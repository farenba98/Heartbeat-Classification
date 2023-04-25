from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Softmax


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
model.add(Dropout(rate=0.5))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(rate=0.5))
model.add(Softmax())
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
