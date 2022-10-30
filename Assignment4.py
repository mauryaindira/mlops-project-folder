
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause
# from statistics import median
# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, tree
from sklearn.model_selection import train_test_split
from skimage import transform
# from tabulate import tabulate

def new_data(data,size):
	new_features = np.array(list(map(lambda img: transform.resize(
				img.reshape(8,8),(size,size),mode='constant',preserve_range=True).ravel(),data)))
	return new_features

digits = datasets.load_digits()
'''
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
'''

user_split = 0.5
# flatten the images
n_samples = len(digits.images)
#data = digits.images.reshape((n_samples, -1))
user_size = 8
data = new_data(digits.data,user_size)
print(" ")
print('For Image Size = '+str(user_size)+'x'+str(user_size)+' and Train-Val-Test Split => '+str(int(100*(1-user_split)))+
	'-'+str(int(50*user_split))+'-'+str(int(50*user_split)))

GAMMA = [0.05,0.003,0.002, 0.004]
C = [0.5,1,2,4]

best_gam = 0
best_c = 0
best_mean_acc=0
best_train=0
best_val=0
best_test=0
table = [['Gamma','C','Training Acc.','Dev Acc.','Test Acc.']]
sum_cal, sum_svm = 0,0
accuracy_val1 =[]
accuracy_training =[]
accuracy_val2 =[]
list_train= [0.5, 0.8, 0.10, 0.9, 0.15]
list_test= [0.5, 0.2, 0.90, 0.1, 0.05]


def  classifier_svm(X_train, X, y_train, y, x_val, x_test, y_val, y_test):
	sum_cal= 0

	for GAM in GAMMA:
		for c in C:
			hyper_params = {'gamma':GAM, 'C':c}
			clf = svm.SVC()
			clf.set_params(**hyper_params)
			clf.fit(X_train, y_train)
			predicted_val = clf.predict(x_val)
			predicted_train = clf.predict(X_train)
			predicted_test = clf.predict(x_test)
			accuracy_val = 100*metrics.accuracy_score(y_val,predicted_val)
			accuracy_train = 100*metrics.accuracy_score(y_train, predicted_train)
			accuracy_test = 100*metrics.accuracy_score(y_test, predicted_test)

			accuracy_val1.append(accuracy_val)
			accuracy_training.append(accuracy_train)
			accuracy_val2.append(accuracy_test)
	for i in range(len(accuracy_val2)):
		sum_cal+= accuracy_val2[i]
	return sum_cal/len(accuracy_val2)


def  classifier_decision_tree(X_train, X, y_train, y, x_val, x_test, y_val, y_test):
	clf = tree.DecisionTreeClassifier()
	clf.fit(X_train, y_train)
	predicted_val = clf.predict(x_val)
	predicted_train = clf.predict(X_train)
	predicted_test = clf.predict(x_test)
	accuracy_val = 100*metrics.accuracy_score(y_val,predicted_val)
	accuracy_train = 100*metrics.accuracy_score(y_train, predicted_train)
	accuracy_test = 100*metrics.accuracy_score(y_test, predicted_test)
	return accuracy_test

for val in range(len(list_train)):
	X_train, X, y_train, y = train_test_split(data, digits.target, test_size=list_train[val],shuffle=False)
	x_val, x_test, y_val, y_test = train_test_split(X,y,test_size=list_test[val],shuffle=False)
	svm_val= classifier_svm(X_train, X, y_train, y, x_val, x_test, y_val, y_test)
	classifier_val= classifier_decision_tree(X_train, X, y_train, y, x_val, x_test, y_val, y_test)
	print("svm: {} , classifier_val: {} ".format(str(svm_val), str(classifier_val)))
	
	sum_cal+= classifier_val
	sum_svm+= svm_val

print("SVM_mean: {} , classifier_mean: {} ".format(str(sum_svm/5), str(sum_cal/5)))
