import numpy as np
import pandas

def load_fer2013(path="."):
	"""
	Method to load the FER2013 dataset and extract the training dataset.

	Args:
		path: String, path to the dataset

	Returns:
		x_train: [-1, 48, 48], training data points
		y_train: [-1, 1], training labels
		x_test: [-1, 48, 48], test data points
		y_test: [-1, 1], test labels
	"""
	# read the dataset as Pandas DataFrame
	csv_path = path + "fer2013.csv"
	data = pandas.read_csv(csv_path)
	# convert DataFrame to numpy array
	data = data.as_matrix()
	# extract x_train, y_train, x_test and y_test
	read_count = 0
	print "Loading dataset.."
	x_train = data[:28709, 1]
	y_train = data[:28709, 0]
	x_test = data[28709:, 1]
	y_test = data[28709:, 0]
	
	# convert the list to numpy array and reshape
	for i in range(x_train.shape[0]):
		x_train[i] = np.array(map(lambda x: int(x), x_train[i].split()))
	x_train = np.column_stack(x_train).T
	for i in range(x_test.shape[0]):
		x_test[i] = np.array(map(lambda x: int(x), x_test[i].split()))
	x_test = np.column_stack(x_test).T

	# normalize the dataset
	x_train = np.divide(np.subtract(x_train, np.mean(x_train, axis=0)), np.std(x_train, axis=0))
	x_test = np.divide(np.subtract(x_test, np.mean(x_test, axis=0)), np.std(x_test, axis=0))
	
	x_train = x_train.reshape((-1, 48, 48, 1))
	x_test = x_test.reshape((-1, 48, 48, 1))
	print "Dataset loaded successfully.."

	return [x_train, y_train, x_test, y_test]

if __name__ == '__main__':
	# to obtain the train and test dataset, use below call
	# [x_train, y_train, x_test, y_test] = load_fer2013(path="fer2013/")
	pass