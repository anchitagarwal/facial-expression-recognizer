from load_data import load_fer2013
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.regularizers import l2

class model_cnn():
	def __init__(self):
		"""
		Initialize the class and build the model.
		"""
		# load the dataset
		[self.x_train, self.y_train, self.x_test, self.y_test] = load_fer2013(path="fer2013/")

		# convert labels to categorical data
		self.y_train = np_utils.to_categorical(self.y_train, num_classes=7)
		self.y_test = np_utils.to_categorical(self.y_test, num_classes=7)

		# build the model
		self.model = self.build_model()

	def build_model(self):
		"""
		Method that builds the model and compiles.

		Returns:
			model: Keras model
		"""
		# define the hyperparameters
		nb_train = self.x_train.shape[0]
		input_dim = self.x_train.shape[1]
		input_channels = self.x_train.shape[3]

		inputs = Input(shape=(input_dim, input_dim, input_channels))
		conv_block_1 = Conv2D(64, (3, 3), padding='same')(inputs)
		conv_block_1 = BatchNormalization()(conv_block_1)
		conv_block_1 = Activation('relu')(conv_block_1)
		conv_block_1 = Conv2D(64, (3, 3), padding='same')(conv_block_1)
		conv_block_1 = BatchNormalization()(conv_block_1)
		conv_block_1 = Activation('relu')(conv_block_1)
		conv_block_1 = MaxPooling2D(pool_size=(2, 2))(conv_block_1)
		conv_block_1 = Dropout(0.5)(conv_block_1)

		conv_block_2 = Conv2D(128, (3, 3), padding='same')(conv_block_1)
		conv_block_2 = BatchNormalization()(conv_block_2)
		conv_block_2 = Activation('relu')(conv_block_2)
		conv_block_2 = Conv2D(128, (3, 3), padding='same')(conv_block_2)
		conv_block_2 = BatchNormalization()(conv_block_2)
		conv_block_2 = Activation('relu')(conv_block_2)
		conv_block_2 = MaxPooling2D(pool_size=(2, 2))(conv_block_2)
		conv_block_2 = Dropout(0.5)(conv_block_2)

		conv_block_3 = Conv2D(256, (3, 3), padding='same')(conv_block_2)
		conv_block_3 = BatchNormalization()(conv_block_3)
		conv_block_3 = Activation('relu')(conv_block_3)
		conv_block_3 = Conv2D(256, (3, 3), padding='same')(conv_block_3)
		conv_block_3 = BatchNormalization()(conv_block_3)
		conv_block_3 = Activation('relu')(conv_block_3)
		conv_block_3 = MaxPooling2D(pool_size=(2, 2))(conv_block_3)
		conv_block_3 = Dropout(0.5)(conv_block_3)

		global_avg_block = Conv2D(7, (3, 3), padding='same', activation='relu')(conv_block_3)
		global_avg_block = GlobalAveragePooling2D()(global_avg_block)

		outputs = Dense(7, activation='softmax', kernel_regularizer=l2(0.01))(global_avg_block)

		# build and compile model
		model = Model(inputs=inputs, outputs=outputs)
		model.summary()
		model.compile(
				optimizer='adadelta',
				loss='hinge',
				metrics=['accuracy'])

		return model

	def train(self):
		"""
		Method that trains the model on training dataset.
		"""
		self.model.fit(x=self.x_train, y=self.y_train,
			epochs=30,
			batch_size=64,
			validation_split=0.15)

	def predict(self):
		"""
		Method that evaluates test data
		"""
		score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
		print score

if __name__ == "__main__":
	model_cnn = model_cnn()
	model_cnn.train()
	score = model_cnn.predict()