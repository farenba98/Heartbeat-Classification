from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# # Define the model
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_shape)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(units=10, activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
