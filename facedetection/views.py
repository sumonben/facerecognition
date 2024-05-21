from pathlib import Path
from django.shortcuts import redirect, render
import cv2
import tensorflow as tf
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from django.http import HttpResponse, JsonResponse
import numpy as np
import urllib.request
import json,os
from keras_facenet import FaceNet
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

embedder=FaceNet()
detector = MTCNN()

def uploadImage(request):
    return render(request,'uploadimage.html')
# Create your views here.
def facedetect(request):
# initialize the MTCNN detector
	#detector = MTCNN()

	# load the input image and convert it to grayscale
	image = cv2.imread('C:/Users/CEDP/Django Project/face/face/media/media/images.jpg')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect the face using MTCNN
	faces = detector.detect_faces(image)

	# draw a bounding box around the face
	for face in faces:
		x, y, w, h = face['box']
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

	# save the image with the bounding box to the output directory
	cv2.imwrite('C:/Users/CEDP/Django Project/face/face/media/media/bounding_box.jpg', image)
 
def facedetectfromvideo(request):
    # initialize the MTCNN detector
	#detector = MTCNN()

	# initialize the video capture object for the default camera
	cap = cv2.VideoCapture(0)

	while True:
		# read the frame from the camera
		ret, frame = cap.read()

		# detect faces using MTCNN
		faces = detector.detect_faces(frame)

		# draw bounding boxes around the faces
		for face in faces:
			x, y, w, h = face['box']
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		# show the resulting frame
		cv2.imshow('Real-time Face Detection', frame)

		# press 'q' key to exit
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# release the video capture object and close all windows
	cap.release()
	cv2.destroyAllWindows()
 

def predict_face(request):
    facenet = FaceNet()
    faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
    Y = faces_embeddings['arr_1']
    encoder = LabelEncoder()
    encoder.fit(Y)
    haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

    if request.method=='POST':
        img=cv2.imread('C:/Users/CEDP/Django Project/face/face/media/media/dataset/Amber_heard/3.jpg')
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        x,y,w,h=detector.detect_faces(img)[0]['box']
        face=img[y:y+h, x:x+w]
        face_arr=cv2.resize(face,(160,160))
        img=get_embedding(face_arr)
        preds_img=[img]
        model.fit(preds_img)
        ypredict=model.predict(preds_img)
    print(f"Face  is:{ypredict}")
    
    return HttpResponse(ypredict)
        
        
def facerecognition(request):
    
    EMBEDDED_X=[]
    faceloading=FACELOADING("C:/Users/CEDP/Django Project/face/face/media/media/dataset")
    X,Y=faceloading.load_clases()
    '''plt.figure(figsize=(18,16))
    for num,image in enumerate(X):
            ncols=3
            nrows=len(Y)//ncols+1
            plt.subplot(nrows,ncols, num+1)
            plt.imshow(image)
            plt.axis('off')'''
    for img in X:
        EMBEDDED_X.append(get_embedding(img))
    EMBEDDED_X=np.asarray(EMBEDDED_X)
    np.savez_compressed('faces_embeddings_done_4classes.npz',EMBEDDED_X,Y)
    print(Y)
    encoder=LabelEncoder()
    encoder.fit(Y)
    Y=encoder.transform(Y)
    #print(Y)

    X_train,X_test,Y_train,Y_test=train_test_split(EMBEDDED_X,Y,shuffle=True,random_state=17)
    model=SVC(kernel='linear',probability=True)
    model.fit(X_train,Y_train)
    ypreds_train=model.predict(X_train)
    ypreds_test=model.predict(X_test)
    print(f"Folder serial:{Y_train,ypreds_train}")

    print(f"Accuracy Score is:{accuracy_score(Y_train,ypreds_train)}")
    print(f"Accuracy Score is:{accuracy_score(Y_test,ypreds_test)}")
    '''t_im=cv2.imread('C:/Users/CEDP/Django Project/face/face/media/media/data/Amber_heard/3.jpg')
    t_im=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    x,y,w,h=detector.detect_faces(t_im)[0]['box']
    t_im=t_im[y:y+h, x:x+w]
    t_im=cv2.resize(t_im,(160,160))
    test_im=get_embedding(t_im)
    preds_img=[test_im]
    ypredict=model.predict(preds_img)
    print(encoder.transform(ypredict))'''
        
    
    
    return HttpResponse("sdmn")

def get_embedding( face_img):
    face_img=face_img.astype('float32')
    face_img=np.expand_dims(face_img, axis=0)
    yhat=embedder.embeddings(face_img)
    return yhat[0]
    




class FACELOADING:
    def __init__(self, directory):
        self.directory=directory
        self.target_size=(160,160)
        self.X=[]
        self.Y=[]
        self.detector=MTCNN()
        
    def extract_face(self, filename):
        img=cv2.imread(filename)
        print(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x,y,w,h=self.detector.detect_faces(img)[0]['box']
        face=img[y:y+h, x:x+w]
        face_arr=cv2.resize(face,self.target_size)
        return face_arr
    
    def load_faces(self, dir):
        FACE=[]
        for im_name in os.listdir(dir):
            try:
                path=dir+im_name
                single_face=self.extract_face(path)
                FACE.append(single_face)
            except Exception as e:
                pass
        
        return FACE
    
    def load_clases(self):
        for sub_dir in os.listdir(self.directory):
            path=self.directory+'/'+sub_dir+'/'
            FACES=self.load_faces(path)
            labels=[sub_dir for _ in range(len(FACES))]
            print(f"Loaded Successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)

def realtime_facerecognition(request):
	facenet = FaceNet()
	faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
	Y = faces_embeddings['arr_1']
	encoder = LabelEncoder()
	encoder.fit(Y)
	Y=encoder.transform(Y)
	haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

	cap = cv2.VideoCapture(0)

	# WHILE LOOP

	while cap.isOpened():
		ret, frame = cap.read()
		rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		#gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = haarcascade.detectMultiScale(rgb_img, 1.3, 5)
		for x,y,w,h in faces:
			img = rgb_img[y:y+h, x:x+w]
			img = cv2.resize(img, (160,160)) # 1x160x160x3
			img = np.expand_dims(img,axis=0)
			ypred = facenet.embeddings(img)
			face_name = model.predict(ypred)
			final_name = encoder.inverse_transform(face_name)
			cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 10)
			cv2.putText(frame, str(final_name), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
					1, (0,0,255), 3, cv2.LINE_AA)

		cv2.imshow("Face Recognition:", frame)
		if cv2.waitKey(1) & ord('q') ==27:
			break

	cap.release()
	cv2.destroyAllWindows
	return HttpResponse("sumon")

