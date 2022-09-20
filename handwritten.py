import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import pandas as pd
from PIL import Image

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
train_frac = 0.1
test_frac = 0.1
dev_frac = 0.1

X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=1-train_frac, shuffle=True
)

X_test,X_dev, y_test,y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/(1-train_frac), shuffle=True
)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
from sklearn.metrics import accuracy_score

gammas = [0.01, 0.005, 0.001, 0.0005]
Cs =     [0.1, 0.2, 0.5, 0.7, 1, 2]
digits = datasets.load_digits()
data = digits.images.reshape((n_samples, -1))
print("---Digits--")
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
print(digits.images[0].size)
df =pd.DataFrame()


pg_gamma = []
pc_gamma =[]
acc_gamma =[]

for gamma in gammas:
    for a in Cs:
        # Create a classifier: a support vector classifier
        clf = svm.SVC(gamma=gamma, C=a)
        
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted = clf.predict(X_dev)
        
        score = accuracy_score(y_pred=predicted,y_true=y_dev)
        
        pg_gamma.append(gamma)
        pc_gamma.append(a)
        acc_gamma.append(score)
              
df['Gamma'] = pg_gamma
df['C']= pc_gamma
df['Accuracy'] = acc_gamma
print(df)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 10))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8,8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    image_resized = resize(image, (int(image.shape[0] // 4), int(image.shape[1] // 2)),
                       anti_aliasing=True)
    


    ax.set_title(f"Prediction  : {prediction}" f"ImageSize: {image.size}")
    print("the image with the size" f"Prediction: {prediction}" f"image_resized :{image_resized}\n")
