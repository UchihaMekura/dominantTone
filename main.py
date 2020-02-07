import os 
import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


def resize(img, target_row):
	row = img.shape[0]
	col = img.shape[1]

	scale = col/row
	target_col = int(target_row * scale)

	img_resized = cv2.resize(img,(target_col,target_row))

	return img_resized


def getCenters(img_bgr):

	img_bgr = resize(img_bgr,150)

	row = img_bgr.shape[0]
	col = img_bgr.shape[1]

	img_bgr = img_bgr.reshape(row*col,3)

	tmp_img = np.array([[0,0,0]])
	for i in range(img_bgr.shape[0]):
		L = max((img_bgr.astype(int)[i]))+min(img_bgr.astype(int)[i])

		if L > 80 and L < 420:
			if (tmp_img == np.array([[0,0,0]])).all():
				tmp_img[0] = img_bgr[i]
			else:
				tmp_img = np.append(tmp_img, [img_bgr[i]],axis = 0)

	img_bgr = tmp_img.copy()

	bandwidth = estimate_bandwidth(img_bgr, quantile=0.2, n_samples=500)
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	ms.fit(img_bgr)

	return ms.cluster_centers_.astype(np.uint8)

def generateSum(img_bgr, centers):

	row = int(img_bgr.shape[0]*0.1)
	col = img_bgr.shape[1]

	#L_list = [max((centers.astype(int)[i]))+min(centers.astype(int)[i]) for i in range(centers.shape[0])]

	result = np.zeros([row, col, 3], dtype = np.uint8)

	centers_number = centers.shape[0]
	
	interval = int(col / centers_number)

	for i in range(result.shape[0]):
		for  j in range(result.shape[1]):
			for k in range(centers_number):
				if j >= k*interval and j <= k*interval + interval:
					result[i][j] = centers[k]



	return result


if __name__ == '__main__':

	folder_src = 'src'
	folder_dst = 'dst'

	file_list = os.listdir(folder_src)

	for each in file_list:
		img_path = folder_src+os.sep+each	
		#rgb_img = cv2.imread(img.encode('gbk').decode(),cv2.IMREAD_UNCHANGED)
		rgb_img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
	
		#b,g,r = cv2.split(rgb_img)
		#rgb_img = cv2.merge((b,g,r))
	
		c = getCenters(rgb_img)
	
		print(c)
		print(c.shape[0])
	
		t = generateSum(rgb_img, c)
	
		result = np.vstack((rgb_img,t))

		cv2.imwrite(folder_dst+os.sep+each,result)
	
		#cv2.imshow('test',resize(result,600))
		#cv2.waitKey(0)

