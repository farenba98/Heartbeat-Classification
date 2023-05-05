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

# Compile the model with the Adam optimizer and specified batch size
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# split_index = int(0.9 * len(X))
# X_train, X_test = X[:split_index], X[split_index:]
# y_train, y_test = y[:split_index], y[split_index:]

kfold = 10
fold_size = len(X) // kfold

acc_train = []
sens_train = []
spec_train = []
CMs_train = []

acc_test = []
sens_test = []
spec_test = []
CMs_test = []

histories = []

for fold in range(kfold):
    print(f"Fold {fold+1}")
    X_train = np.concatenate((X[:fold*fold_size], X[(fold+1)*fold_size:]))
    X_test = X[fold*fold_size:(fold+1)*fold_size]
    y_train = np.concatenate((y[:fold*fold_size], y[(fold+1)*fold_size:]))
    y_test = y[fold*fold_size:(fold+1)*fold_size]

    history = model.fit(x=X_train, 
            y=y_train, 
            epochs=20,
            batch_size=256,
            validation_data=(X_test, y_test),
            callbacks=[tensorboard_callback])
    
    histories.append(history)

    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=256)

    y_pred_train = model.predict(X_train)
    y_pred_train = np.argmax(y_pred_train, axis=1)
    y_pred_test = model.predict(X_test)
    y_pred_test = np.argmax(y_pred_test, axis=1)
 
    # Calculate the confusion matrix
    cm_train = np.array(tf.math.confusion_matrix(np.argmax(y_train, axis=1), y_pred_train))
    cm_test = np.array(tf.math.confusion_matrix(np.argmax(y_test, axis=1), y_pred_test))

    # calculate tp, tn, fp, and fn
    tp_train = np.diag(cm_train)
    tn_train = np.sum(cm_train) - np.sum(cm_train, axis=0) - np.sum(cm_train, axis=1) + tp_train
    fp_train = np.sum(cm_train, axis=0) - tp_train
    fn_train = np.sum(cm_train, axis=1) - tp_train

    # calculate accuracy, sensitivity, and specificity
    accuracy_train = (tp_train + tn_train) / np.sum(cm_train)
    sensitivity_train = tp_train / (tp_train + fn_train)
    specificity_train = tn_train / (tn_train + fp_train)
        
    acc_train.append(accuracy_train)
    sens_train.append(sensitivity_train)
    spec_train.append(specificity_train)
    CMs_train.append(cm_train)

    # calculate tp, tn, fp, and fn
    tp_test = np.diag(cm_test)
    tn_test = np.sum(cm_test) - np.sum(cm_test, axis=0) - np.sum(cm_test, axis=1) + tp_test
    fp_test = np.sum(cm_test, axis=0) - tp_test
    fn_test = np.sum(cm_test, axis=1) - tp_test
    # calculate accuracy, sensitivity, and specificity
    accuracy_test = (tp_test + tn_test) / np.sum(cm_test)
    sensitivity_test = tp_test / (tp_test + fn_test)
    specificity_test = tn_test / (tn_test + fp_test)

    acc_test.append(accuracy_test)
    sens_test.append(sensitivity_test)
    spec_test.append(specificity_test)
    CMs_test.append(cm_test)

print("\nAccuracy:\n", acc_train)
print("\nSensitivity:\n", sens_train)
print("\nSpecificity:\n", spec_train)

print("\nAccuracy:\n", acc_test)
print("\nSensitivity:\n", sens_test)
print("\nSpecificity:\n", spec_test)

best_acc_index = np.argmax(np.mean(acc_test))
best_cm_train = CMs_train[best_acc_index]
best_cm_test = CMs_test[best_acc_index]

history = histories[best_acc_index]

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(best_cm_train, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('CM for Train Data')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(best_cm_test, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('CM for Test Data')
plt.show()

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()