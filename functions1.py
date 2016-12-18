from libraries import *
from variants import *
from finalres import *
def sliding_window(image,stepsize,windowsize):
    for y in range(0,image.shape[0],stepsize[0]):
        for x in range(0,image.shape[1],stepsize[1]):
            yield(x,y,image[y:y+windowsize[1],x:x+windowsize[0]])
def pyramid(image,imag,pic,scale=1.5,minsize=(40,40)): 
	sc=0                            
	yield (image,imag,pic,sc)
	while True:
		sc+=1
		w=int(image.shape[1]/scale)
		image=imutils.resize(image,width=w)
		imag=imutils.resize(imag,width=w)
		if image.shape[0]<minsize[1] or image.shape[1]<minsize[0]:   
			break
		yield (image,imag,pic,sc)
def nms(detections):
	detections = sorted(detections, key=lambda detections: detections[2],reverse=True)
	pick=[]
	pick.append(detections[0])
	return pick
def preprocess_img(img):
	hsv = color.rgb2hsv(img)
	hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
	img  = color.hsv2rgb(hsv)
	min_side = min(img.shape[:-1])
	centre = img.shape[0]//2, img.shape[1]//2
	img = img[centre[0]-min_side//2:centre[0]+min_side//2,
	centre[1]-min_side//2:centre[1]+min_side//2,
		      :]
	img = transform.resize(img, (40,40))
	img = np.rollaxis(img,-1)
	return img
def dofunction(img,imag,pic,sc):                                      
	clone=copy.copy(img)
	clone2=copy.copy(imag)
	#cv2.resize(img,(200,150))
	if img.shape[0]<wdw_sz[1] or img.shape[1]<wdw_sz[0]:
		return
	for (x,y,swh) in sliding_window(img,step_size,wdw_sz):
		if swh.shape[0]!=wdw_sz[1] or swh.shape[1]!=wdw_sz[0]:
			continue
		list_fd=[]
		fd=hog(swh, orientations=8, pixels_per_cell=(5,5),cells_per_block=(2,2), visualise=False)
		list_fd.append(fd)
		hog_features = np.array(list_fd, 'float64')
		pred=clf.predict(hog_features)
		prob=clf.decision_function(hog_features)	
		if pred==1:# and prob>0.3:		
			pred2=clf2.predict(hog_features)
			prob=clf2.decision_function(hog_features)
			if pred2==1 and prob>0.71:
				print(prob)
				y1=int(y*(downscale**sc))
				x1=int(x*(downscale**sc))
				w1=int(40*(downscale**sc))
				crop_img = pic[y1:y1+w1,x1:x1+w1]
				cv2.imwrite("/home/saloni/detect/w.jpg",crop_img)
				im=Image.open("/home/saloni/detect/w.jpg")
				im.save("/home/saloni/detect/Optimized-new11.jpg",dpi=(96,96))
				crop_img=cv2.imread("/home/saloni/detect/Optimized-new11.jpg")

				im.save("/home/saloni/detect/new1w.jpg",dpi=(96,96))
				im=cv2.imread("/home/saloni/detect/new1w.jpg")
				crop_img = preprocess_img(crop_img)

				X_test=[]
				X_test.append(crop_img)
				X_test = np.array(X_test)
				'''json_file = open('model143.json', 'r')
	
				loaded_model_json = json_file.read()
				json_file.close()
				loaded_model = model_from_json(loaded_model_json)
				loaded_model.load_weights("model143.h5")
				loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])'''
				y_pred = loaded_model.predict_classes(X_test)
				prob= loaded_model.predict_proba(X_test)
				print(prob[0][y_pred[0]])
				if prob[0][y_pred[0]]>0.85:
					#cv2.imshow("CROPPED IMAGE",crop_img)
					#cv2.moveWindow("CROPPED IMAGE",1000,200)
					#cv2.waitKey(time)
					#cv2.destroyAllWindows()		
					output(y_pred[0])
					cv2.rectangle(clone2,(x,y),(x+40,y+40),(0,255,0),2)
					#classifier(clone2,x,y,40)
					cv2.imshow("a",clone2)
					cv2.moveWindow("a",250,100)
					cv2.waitKey(time)
					cv2.destroyAllWindows()
'''def classifier(img,x,y,w):
	crop_img = img[y:y+w,x:x+w]
	cv2.imshow("CROPPED IMAGE",crop_img)
	cv2.moveWindow("CROPPED IMAGE",1000,200)
	cv2.waitKey(time)
	cv2.destroyAllWindows()
	
	cv2.imwrite("/home/saloni/detect/w.jpg",crop_img)
	im=Image.open("/home/saloni/detect/w.jpg")
	im.save("/home/saloni/detect/Optimized-new11.jpg",dpi=(96,96))
	crop_img=cv2.imread("/home/saloni/detect/Optimized-new11.jpg")

	im.save("/home/saloni/detect/new1w.jpg",dpi=(96,96))
	im=cv2.imread("/home/saloni/detect/new1w.jpg")
	crop_img = preprocess_img(crop_img)

	X_test=[]
	X_test.append(crop_img)
	X_test = np.array(X_test)
	json_file = open('model143.json', 'r')
	
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model143.h5")
	loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	y_pred = loaded_model.predict_classes(X_test)
	prob= loaded_model.predict_proba(X_test)
	print(prob[0][y_pred[0]])
	if prob[0][y_pred[0]]>0.85:		
		output(y_pred[0])
	return

'''
