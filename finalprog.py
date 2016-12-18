from libraries import *
from functions1 import *
from variants import *
#from FINALRESULT import *
from finalres import *



#image=io.imread(image)
#im=color.rgb2gray(image)


#clone2=copy.copy(im)

#for image in glob.glob("/home/saloni/testimgs2/*"):
#for image in sorted(glob.glob("/home/saloni/testimgs2/*.ppm")):
for image in sorted(glob.glob("/home/saloni/final/imgs1/*")):
	pic=io.imread(image)
	#pic=cv2.resize(pic,(700,411))
	pic=cv2.resize(pic,(1020,600))
	im=color.rgb2gray(pic)
	Parallel(n_jobs=100)(delayed(dofunction)(img,imag,pic,sc)  for img,imag,pic,sc in pyramid(im,pic,pic))

