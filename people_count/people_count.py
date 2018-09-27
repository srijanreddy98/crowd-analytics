import face_recognition
import cv2
import os
import time
import numpy as np
import pickle
import _pickle as cPickle

known_face_names = []
known_face_encodings = []

path = os.getcwd()+'\\' + 'people_count\\'

def modelLoad():
	with tf.Session as sess:
		tf.Saver.restore('vgg-face')

def process_encode():
	global known_face_encodings
	global known_face_names
	if os.path.getsize(path + 'face_encodings.pkl'):
		with open(path + "face_encodings.pkl", "rb") as fp:  # Unpickling
			unpickler = pickle.Unpickler(fp)
			known_face_encodings = unpickler.load()
			# known_face_encodings = pickle.load(fp)
		with open(path + "face_names.pkl", "rb") as fp_:  # Unpickling
			unpickler = pickle.Unpickler(fp_)
			known_face_names = pickle.load(fp_)
			# print(known_face_names)
 

def process() :
	global known_face_names
	global known_face_encodings
	known_face_names = []
	known_face_encodings = []
	datasets = os.listdir('./datasets')
	ckp = modelLoad()
	for i in datasets:
		temp_name = str(i)
		i = './datasets/' + i

		imag = face_recognition.load_image_file(i)
		encod = face_recognition.face_encodings(imag)[0]
		# print(len(encod))
		known_face_encodings.append(encod)

		temp_name = temp_name.split('.')
		# print(temp_name)
		name = str(temp_name[0])
		known_face_names.append(name)

def add_to_known(frame) :
	global known_face_names
	global known_face_encodings
	img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, 1)

	# imag = face_recognition.load_image_file(frame)
	encod = face_recognition.face_encodings(frame)

	# print("Length of encod = ",len(encod))
	# print(encod)
	# print('\n')

	known_face_encodings.append(encod)

	# print(known_face_encodings[-1])
	# print('\n')

	# print(known_face_encodings[-2])

	file_count = len(known_face_names)
	# print(file_count)

	# print(temp_name)
	name = str(file_count)
	known_face_names.append(name)
	# print(known_face_names)


def predict():
	# print(known_face_names)
	people_count = len(known_face_encodings)
	# video_capture = cv2.VideoCapture(0)
	f = open('face_encodings.pkl', 'ab')
	f_ = open('face_names.pkl', 'ab')
	pickler = cPickle.Pickler(f)
	pickler_ = cPickle.Pickler(f_)
	# Initialize some variables

	# print(known_face_names)
	face_locations = []
	face_encodings = []
	face_names = []
	process_this_frame = True
	time_to_predict = 1000   #Time it takes to predict
	NI_count = 0
	ucnt = 0
	ncnt = 0
	fin_name = ''
	start = time.time()
	while True:
		try:
			# print(people_count)
			# print("Number of people : ", len(known_face_encodings))
			faceT = {"len": len(known_face_encodings)}
			import json
			import codecs
			with open('4forces3.json', 'wb') as f:
				json.dump(faceT, codecs.getwriter('utf-8')(f), ensure_ascii=False)
			testp = path + '../FINAL/src/3.txt'
			print(os.path.exists(testp))
			if(os.path.exists(testp)):
				frame = cv2.imread(path+'../FINAL/src/1.jpg')
			else:
				frame = cv2.imread(path+'../FINAL/src/2.jpg')
			# frame = cv2.imread('images.jpg')
			small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
			rgb_small_frame = small_frame[:, :, ::-1]
			# cv2.imshow("temp",rgb_small_frame)
			if process_this_frame:
				# Find all the faces and face encodings in the current frame of video
				face_locations = face_recognition.face_locations(rgb_small_frame)
				face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

				# print(face_locations)

				face_names = []
				for face_encoding in face_encodings:
					matches = face_recognition.compare_faces(known_face_encodings, face_encoding , tolerance=0.6)
					name = "Unknown"

					if True in matches:
						first_match_index = matches.index(True)
						name = known_face_names[first_match_index]
					else:
						people_count += 1
						# print(known_face_names)
						# add_to_known(frame)
						known_face_encodings.append(face_encoding)
						pickler.dump(face_encoding)
						known_face_names.append(len(known_face_names))
						pickler_.dump(len(known_face_names))




			process_this_frame = not process_this_frame


			# Display the results
			for (top, right, bottom, left), name in zip(face_locations, face_names):
				# Scale back up face locations since the frame we detected in was scaled to 1/4 size
				top *= 4
				right *= 4
				bottom *= 4
				left *= 4

				# Draw a box around the face
				cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)

				# Draw a label with a name below the face
				# cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255))
				font = cv2.FONT_HERSHEY_DUPLEX
				cv2.putText(frame, name, (left + int((right-left)/2), bottom + 16), font, 0.5, (255, 255, 255), 1)

			# Display the resulting image
			# cv2.imshow('Video', frame)
			if len(face_names) == 1:
				if name == 'Unknown':
					ucnt += 1
				else :
					ncnt += 1

				if len(face_names)==1:
					# print("Predicted user : ",face_names[0] )
					fin_name = face_names[0]
			elif len(face_names)>1 :
				ucnt = 0
				ncnt = 0
			# print(time.time() - start)
			if time.time() - start >= time_to_predict:
				break
			# Hit 'q' on the keyboard to quit!
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		except:
			print('except')
			continue
	# Release handle to the webcam
	# video_capture.release()
	# cv2.destroyAllWindows()
	# if(ncnt > ucnt):
	#^ print("FINAL PREDICTION: ",fin_name)
	#^ print("Accuracy = ",ncnt/(ncnt+ucnt)*100,"%")
	# elif len(face_names) >= 1:
	#^ print('Too many faces in the frame')
	# else :
	#^ print("Could Not Identify face")

# process()
# process_encode()
predict()
