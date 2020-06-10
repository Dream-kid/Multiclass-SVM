# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import GaussianNB
from sklearn.naive_bayes import GaussianNB
import cv2
# import warnings to remove any type of future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def func():
	# reading csv file and extracting class column to y.
	dataf = pd.read_csv("Datasetinfectedhealthy.csv")

	# extracting two features
	X = dataf.drop(['imgid','fold num'], axis=1)
	y = X['label']
	X = X.drop('label', axis=1)

	print("\nTraining dataset:-\n")
	print(X)


	log = pd.read_csv("datasetlog/Datasetunlabelledlog.csv")

	log = log.tail(1)
	X_ul = log.drop(['imgid','fold num'], axis=1)

	print("\nTest dataset:-\n")
	print(X_ul)

	print("\n*To terminate press (q)*")

	#X.plot(kind='scatter',x='feature1',y='feature2')
	#plt.show()


	Sum = 0

	from sklearn.model_selection import train_test_split

	'''
	from sklearn.model_selection import train_test_split  
	for n in range(4):
		x_train, Xi_test, y_train, yi_test = train_test_split(X, y, test_size=0.52, random_state=60)  
		if cv2.waitKey(1) == ord('q' or 'Q'): break     
		svclassifier = SVC(kernel='linear')  
		svclassifier.fit(x_train, y_train)  
		pred = svclassifier.predict(X_ul)
	'''
	for n in range(4):
		x_train, Xi_test, y_train, yi_test = train_test_split(X, y, test_size=0.52, random_state=60)
		if cv2.waitKey(1) == ord('q' or 'Q'): break
		classifier = GaussianNB()
		classifier.fit(x_train, y_train)
		pred =classifier.predict(X_ul)
		Sum = Sum + pred
		print(pred)

	print("\nprediction: %d" %int(Sum/4))

	if(Sum < 2):

		print("The leaf is sufficiently healthy!")
		return 1
	else:
		print("The leaf is infected!")
		return 0



