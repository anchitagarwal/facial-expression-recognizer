from SimpleCV.Features import HueHistogramFeatureExtractor, EdgeHistogramFeatureExtractor, HaarLikeFeatureExtractor
from SimpleCV.MachineLearning import SVMClassifier
from SimpleCV.ImageClass import ImageSet
from SimpleCV.Color import Color
from load_data import load_fer2013
import numpy as np
import random

class Trainer():
	def __init__(self,classes, trainPaths):
		self.classes = classes
		self.trainPaths = trainPaths

	def getExtractors(self):
		hhfe = HueHistogramFeatureExtractor(10)
		ehfe = EdgeHistogramFeatureExtractor(10)
		haarfe = HaarLikeFeatureExtractor(fname='haar.txt')
		return [hhfe,ehfe,haarfe]

	def getClassifiers(self,extractors):
		props ={
			'KernelType':'RBF', #default is a RBF Kernel
		    'SVMType':'C',     #default is C
		    'nu':None,          # NU for SVM NU
		    'c':100000,           #C for SVM C - the slack variable
		    'degree':None,      #degree for poly kernels - defaults to 3
		    'coef':None,        #coef for Poly/Sigmoid defaults to 0
		    'gamma':None,       #kernel param for poly/rbf/sigma
		}
		svm = SVMClassifier(extractors, props)
		return svm

	def train(self):
		self.svm = self.getClassifiers(self.getExtractors())
		self.svm.train(self.trainPaths,self.classes,verbose=True)

	def test(self,testPaths):
		print self.svm.test(testPaths,self.classes,verbose=True)

	def visualizeResults(self,classifier,imgs):
		for img in imgs:
			className = classifier.classify(img)
			img.drawText(className,10,10,fontsize=60,color=Color.BLUE)		 
		imgs.show()

classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main():
	trainPaths = ['./data_augmentation/sample_img/'+c+'/train/' for c in classes ]
	testPaths =  ['./data_augmentation/sample_img/'+c+'/test/'   for c in classes ]

	trainer = Trainer(classes,trainPaths)
	trainer.train()
	tree = trainer.svm
	
	imgs = ImageSet()
	for p in testPaths:
		imgs += ImageSet(p)
	random.shuffle(imgs)

	print "Result test"
	trainer.test(testPaths)

	trainer.visualizeResults(tree,imgs)

main()

# # build the model
# classifier = svm.SVC(decision_function_shape='ovo')

# # train the model
# print "``fit`` started.."
# classifier.fit(x_train, y_train)
# print "completed.."

# # predict test data
# print "predicting .."
# scores = classifier.predict(x_test)
# import pdb
# pdb.set_trace()