from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

#create model
def create_model(input_shape):
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(input_shape), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(number_of_classes, activation='softmax'))

	return model

def prep_process_data(X_train, X_test, y_train, y_test, number_of_classes):
	# Reshaping to CNN format
	X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
	X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

	# Making sure that the values are float so that we can get decimal points after division
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	# Normalizing the RGB codes by dividing it to the max RGB value.
	X_train/=255
	X_test/=255

	# one hot encode
	y_train = np_utils.to_categorical(y_train, number_of_classes)
	y_test = np_utils.to_categorical(y_test, number_of_classes)

	return (X_train, y_train), (X_test, y_test)


number_of_classes = 10

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train, y_train), (X_test, y_test) = prep_process_data(X_train, X_test, y_train, y_test, number_of_classes)

input_shape = (X_train.shape[1], X_train.shape[2], 1)

# create model
model = create_model(input_shape)

# Compile model
model.compile(loss='categorical_crossentropy', 
				optimizer=Adam(), 
				metrics=['accuracy'])

# Fit model
model.fit(X_train, y_train, 
			validation_data=(X_test, y_test), 
			epochs=10,
			batch_size=200)

# Save model
model.save('model/model.h5')

# Final evaluation of the model
metrics = model.evaluate(X_test, y_test)
print("[Loss, Accuracy]")
print(metrics)
