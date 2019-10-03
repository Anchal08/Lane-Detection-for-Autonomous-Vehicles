import cv2
import glob
import numpy as np
import pandas as pd
import json
file_path = 'bdd100k_images\\bdd100k\\images\\100k\\val\\'

with open('data_val.json', 'r') as fp:
    data = json.load(fp)

#print(data)



#images = [cv2.imread(file) for file in glob.glob(file_path+"*.jpg")]
files = glob.glob(file_path+"*.jpg")
#print(files[1])
# x = files[1].split("\\")
# print(x[-1])
# image = cv2.imread(files[1])

# for key,value in data.items():
# 	if x[-1] in value:
# 		print(x[-1],value)


for i in range(len(files)):

	x = files[i].split("\\")
	print("i",i)
	vert=[]
	for key,value in data.items():

		if x[-1] in value:
			#print(x[-1],value)
			#print(key.split("_"))
			key_num = key.split("_")
			poly_points = data[str('vertex_')+str(key_num[1])]

			if (len(poly_points) > 0):
				print(poly_points)
				for p in range(len(poly_points)):
					poly = tuple(poly_points[p])
					vert.append(poly)
				#print(vert)

				vert = np.array([vert], dtype=np.int32)
				vert =  vert/2
				vert = np.array([vert], dtype=np.int32)
				#print(vert)
				#print(poly_points)
				#print(type(poly_points))
				image = cv2.imread(files[i])
				#print(image.shape)
				#cv2.imshow("original",image)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				#print(image.shape)
				#cv2.imshow("Gray",image)
				#print(image.shape[0])
				resized_image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
				#print(resized_image.shape)
				#cv2.waitKey(0)
				mask = np.zeros(resized_image.shape)


				cv2.fillPoly(mask,vert,255)

				#cv2.imshow("Image",image)
				#cv2.imshow("Mask",mask)
				cv2.imwrite(str('val_data')+'\\Orig_'+str(x[-1])+'.jpg',image)
				cv2.imwrite(str('val_label')+'\\Mask_'+str(x[-1])+'.jpg',mask)
				cv2.waitKey(0)



	#cv2.imshow("Mask",mask)
	#v2.waitKey(0)