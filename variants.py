from libraries import *


clf=joblib.load("hog_linear70k.pkl")
clf2=joblib.load("hard_mining30k.pkl")
json_file = open('model143.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model143.h5")
loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


downscale=1.5
wdw_sz=(40,40)
step_size=(3,3)
visualize_det=True
detections=[]
scale=0
time=1000
