from sklearn.externals import joblib 
from sklearn import datasets 
from skimage.feature import hog 
from sklearn.svm import LinearSVC 
import numpy as np

from skimage import color, exposure, transform


#features = np.array(dataset.data, 'int16') 
#labels = np.array(dataset.target, 'int')



from skimage import io
import os
import glob

def get_class(img_path):
	return int(img_path.split('/')[-2])


root_dir = '/home/saloni/detect/train/1/'
imgs1 = []
labels1 = []

all_img_paths = glob.glob(os.path.join(root_dir, '*.ppm'))
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
	img = color.rgb2gray(io.imread(img_path))
	img = transform.resize(img, (40,40))

	label = get_class(img_path)
	imgs1.append(img)
	labels1.append(label)

root_dir = '/home/saloni/detect/train/-1'
imgs2 = []
labels2 = []

all_img_paths1 = glob.glob(os.path.join(root_dir, '*.jpg'))
#np.random.shuffle(all_img_paths)
for img_path in all_img_paths1:
	img = color.rgb2gray(io.imread(img_path))
	img = transform.resize(img, (40,40))

	label = get_class(img_path)
	imgs2.append(img)
	labels2.append(label)
imgs = imgs1 + imgs2
labels = labels1 + labels2
X = np.array(imgs, dtype='float32')
# Make one hot targets
#Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
Y=np.asarray(labels);
print X.shape
print Y.shape
#print Y.shape[1]



#Next, we calculate the HOG features for each image in the database and save them in another numpy array named hog_feature.

list_hog_fd = []
for feature in X:
    fd = hog(feature.reshape((40, 40)), orientations=8, pixels_per_cell=(5,5),cells_per_block=(2,2), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

#saving features to text files
#np.savetxt('hog_features_1563.txt',hog_features)
#np.savetxt('labels.txt',Y)

#training a linear svm
#hog_features=np.loadtxt('test123.txt')
#labels=np.loadtxt('labels.txt')

clf = LinearSVC()
clf.fit(hog_features, Y)
#joblib.dump(clf, "svm_linear.pkl", compress=3)


#testing

root_dir = '/home/saloni/detect/Test/'
imgs = []
all_img_paths = glob.glob(os.path.join(root_dir, '*.ppm'))
for img_path in all_img_paths:
	img = color.rgb2gray(io.imread(img_path))
	img = transform.resize(img, (40,40))
	imgs.append(img)
X_test=np.asarray(imgs)
test_hog_fd = []
for feature in X_test:
    fd = hog(feature.reshape((40, 40)), orientations=8, pixels_per_cell=(5,5),cells_per_block=(2,2), visualise=False)
    test_hog_fd.append(fd)
hog_test = np.array(test_hog_fd, 'float64')
'''
print hog_test.shape[0]
print hog_test.shape[1]
#print hog_test.shape[2]
'''

hog_test=hog_test.reshape(hog_test.shape[0],hog_test.shape[1])
predicted=clf.predict(hog_test)
print(predicted)

