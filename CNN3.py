import numpy as np
from skimage import color, exposure, transform
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import model_from_json
import h5py

NUM_CLASSES = 43
IMG_SIZE = 40

def preprocess_img(img):
# Histogram normalization in v channel
	hsv = color.rgb2hsv(img)
	hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
	img = color.hsv2rgb(hsv)

	# central square crop
	min_side = min(img.shape[:-1])
	centre = img.shape[0]//2, img.shape[1]//2
	img = img[centre[0]-min_side//2:centre[0]+min_side//2,
	centre[1]-min_side//2:centre[1]+min_side//2,
		      :]

	# rescale to standard size
	img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

	# roll color axis to axis 0
	img = np.rollaxis(img,-1)

	return img

from skimage import io
import os
import glob

def get_class(img_path):
	return int(img_path.split('/')[-2])

root_dir = '/home/saloni/GTSRB/Final_Training/Images/'
imgs = []
labels = []

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
	img = preprocess_img(io.imread(img_path))
	label = get_class(img_path)
	imgs.append(img)
	labels.append(label)

X = np.array(imgs, dtype='float32')
# Make one hot targets
Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

print X.shape
print Y.shape
#np.savetxt('43imgs.txt',X)
#np.savetxt('43labels.txt',Y)

seed = 7
np.random.seed(seed)

def cnn_model():
	model = Sequential()

	model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(3, IMG_SIZE, IMG_SIZE), activation='relu'))
	model.add(Convolution2D(32, 5, 5, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Convolution2D(64, 5, 5, border_mode='valid', activation='relu'))
	model.add(Convolution2D(64, 5, 5, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	'''model.add(Convolution2D(128, 5, 5, border_mode='valid', activation='relu'))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))'''

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(NUM_CLASSES, activation='softmax'))
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

	return model

#from keras.optimizers import SGD

model = cnn_model()




import pandas as pd
test = pd.read_csv('GT-final_test.csv',sep=';')



# Load test dataset
X_test = []
y_test = []
i = 0
for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
	img_path = os.path.join('/home/saloni/GTSRB/Final_Test/Images/',file_name)
	X_test.append(preprocess_img(io.imread(img_path)))
	y_test.append(class_id)
#
X_test = np.array(X_test)
y_test = np.array(y_test)
y_test = np_utils.to_categorical(y_test)

print X_test.shape
print y_test.shape
#np.savetxt('testimgs.txt',X_test)
#np.savetxt('testlabels.txt',y_test)

# fitting/training the model node)
  #File "/home/saloni/.local/lib/python2.7/site-packages/theano/gof/opt.py", line 1659, in warn
    #raise exc

batch_size = 200
nb_epoch = 10
model.fit(X,Y,validation_data=(X_test, y_test),nb_epoch=nb_epoch,batch_size=batch_size,verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

#serialize model to JSON
model_json = model.to_json()
with open("model43.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model43.h5")

# later...
 
# load json and create model
json_file = open('model43.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model43.h5")
print("Loaded model from disk")


# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print ("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
print("Saved model to disk")
 

# predict and evaluate
#y_pred = model.predict_classes(X_test)
#acc = np.sum(y_pred==y_test)/np.size(y_pred)
#print("Test accuracy = {}".format(acc)) 

# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Baseline Error: %.2f%%" % (100-scores[1]*100))
