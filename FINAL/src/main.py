from statistics import mode

import cv2
from keras.models import load_model
import numpy as np
import os
from req.datasets import get_labels
from req.inference import detect_faces
from req.inference import draw_text
from req.inference import draw_bounding_box
from req.inference import apply_offsets
from req.inference import load_detection_model
from req.preprocessor import preprocess_input


#---------------------------- ROHAN'S CODE ---------------------------------

import tensorflow as tf
import sys
import glob
path = os.getcwd()+'\\' + 'FINAL\\src\\'
# path = os.path.join(path, 'FINAL','src')
face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(path+ 'haarcascade_eye.xml')
print(path + '/haarcascade_frontalface_default.xml',
      path + '/haarcascade_eye.xml')
face = []
IM_SIZE = 32
BATCH_SIZE = 100
WINDOW_SIZE = 2

sess = tf.InteractiveSession()
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x_tensor = tf.placeholder(tf.float32, shape=[None,IM_SIZE,IM_SIZE,3])
y_ = tf.placeholder(tf.float32, shape=[None,2])
keep_prob = tf.placeholder(tf.float32)

x_reshape = tf.reshape(x_tensor, [-1,IM_SIZE,IM_SIZE,3])

W_conv1 = weight_variable([5,5,3,16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_reshape, W_conv1) + b_conv1)

W_conv2 = weight_variable([5,5,16,16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

W_conv3 = weight_variable([5,5,16,16])
b_conv3 = bias_variable([16])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)


"""
1 pooling
"""
h_pool3 = max_pool_2x2(h_conv3)


W_conv4 = weight_variable([5,5,16,32])
b_conv4 = bias_variable([32])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

W_conv5 = weight_variable([5,5,32,32])
b_conv5 = bias_variable([32])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

W_conv6 = weight_variable([5,5,32,32])
b_conv6 = bias_variable([32])
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)


"""
2 pooling
"""
h_pool6 = max_pool_2x2(h_conv6)

W_conv7 = weight_variable([5,5,32,64])
b_conv7 = bias_variable([64])
h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7) + b_conv7)

W_conv8 = weight_variable([5,5,64,64])
b_conv8 = weight_variable([64])
h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8)

W_conv9 = weight_variable([5,5,64,64])
b_conv9 = bias_variable([64])
h_conv9 = tf.nn.relu(conv2d(h_conv8, W_conv9) + b_conv9)


"""
3 pooling
"""
h_pool9 = max_pool_2x2(h_conv9)
h_pool9_flat = tf.reshape(h_pool9, [-1, int((IM_SIZE/8)*(IM_SIZE/8)*64)])



W_fc1 = weight_variable([int((IM_SIZE/8)*(IM_SIZE/8)*64), 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool9_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024,1024])
b_fc2 = bias_variable([1024])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([1024,2])
b_fc3 = bias_variable([2])

y_matmul = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
y_conv = tf.nn.softmax(y_matmul)

l2_loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv+1e-7), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-3).minimize(l2_loss)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.global_variables_initializer().run()

saver = tf.train.Saver()
SAVE_PATH = path + "/checkpoint/ver1.0_iteration.64000.ckpt"
saver.restore(sess, SAVE_PATH)

prev_face = [(0,0,30,30)]
prev_eyes = [(1,1,1,1), (1,1,1,1)]
drowsiness_check_list = [0] * WINDOW_SIZE
drowsiness_check_idx = 0

def atten(eyes,roi_color):
    eye_count = 1
    attentive = False
    global drowsiness_check_idx
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        eye_image = roi_color[ey:ey+eh , ex:ex+ew]
        input_images = cv2.resize(eye_image, (32,32))
        input_images.resize((1,32,32,3))
        input_images = np.divide(input_images, 255.0)
        label = sess.run(tf.argmax(y_conv, 1), feed_dict={keep_prob:1.0, x_tensor:input_images})
        drowsiness_check_list[drowsiness_check_idx%WINDOW_SIZE] = label[0]
        drowsiness_check_idx += 1
        if eye_count == 2:
            if drowsiness_check_list == [1] * WINDOW_SIZE:
                print("Face - "+ str(face_count)+ " - Not Attentive",)
                draw_text(face_coordinates, rgb_image, "Not Attentive",
                          color, 1, -65, 1, 1)
            elif drowsiness_check_list == [0] * WINDOW_SIZE:
                print("Face - "+ str(face_count)+ " - Attentive")
                draw_text(face_coordinates, rgb_image, "Attentive",
                          color, 1, -65, 1, 1)
                attentive = True
        eye_count+=1
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        # cv2.imshow('window_frame', bgr_image)
    return attentive


#--------------------------------- END ------------------------------------




# gender_model_path = path+'FINAL/trained_models/gender_model_vgg.hdf5'
gender_model_path = path + '/../trained_models/gender_model_vgg.hdf5'
gender_labels = get_labels('imdb')

detection_model_path = path + '/../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = path + '/../trained_models/16_layer_relu_test.hdf5'
emotion_labels = get_labels('fer2013')



# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)
emotion_offsets = (20, 40)


# loading models



face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

gender_classifier = load_model(gender_model_path, compile=False)



# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []
gender_window = []


# starting video streaming
# cv2.namedWindow('window_frame')
# video_capture = cv2.VideoCapture(0)
while True:
    try:
        testp = path +'3.txt'
        print(os.path.exists(testp))
        
        if(os.path.exists(testp)):
            bgr_image = cv2.imread(path+'1.jpg')
        else:
            bgr_image = cv2.imread(path+'2.jpg')
        # bgr_image = cv2.imread(path+'/photo.jpg')
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(face_detection, gray_image)
        print(len(faces))

        face = []
        face2 = []
        face_count = 0


        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)

            rgb_face = rgb_image[y1:y2, x1:x2]

            bgr_face = bgr_image[y1:y2, x1:x2]

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                rgb_face = cv2.resize(rgb_face, (gender_target_size))
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]

            rgb_face = np.expand_dims(rgb_face, 0)
            rgb_face = preprocess_input(rgb_face, False)
            gender_prediction = gender_classifier.predict(rgb_face)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = gender_labels[gender_label_arg]


            gender_window.append(gender_text)


            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
                gender_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
                gender_mode = mode(gender_window)
            except:
                continue

            if gender_text == gender_labels[0]:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)


            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                    color, 0, -45, 1, 1)

            draw_text(face_coordinates, rgb_image, gender_mode,
                    color, 0, -20, 1, 1)


            #ROHAN'S CODE
            roi_gray = gray_image[y1:y2, x1:x2]
            roi_color = bgr_image[y1:y2, x1:x2]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            face_count += 1
            text = "Person" + str(face_count)
            # cv.putText(img, text, (int(x + w/2), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cordi = ""
            l = 0
            for cor in face_coordinates:
                if l == 0:
                    cordi += str(cor.item())
                    l += 1
                else:
                    cordi += ','+str(cor.item())
            face.append({"face": cordi, "emotion": emotion_text, "gender": gender_text, "attentive": atten(eyes, roi_color)})
            face2.append({"face": cordi, "attentive": atten(eyes, roi_color)})
            import json
            import codecs
            with open(path + '4forces.json', 'wb') as f:
                # print('here')
                json.dump(face, codecs.getwriter('utf-8')(f), ensure_ascii=False)
            with open(path + '4forces2.json', 'wb') as f:
                # print('here')
                json.dump(face2, codecs.getwriter('utf-8')(f), ensure_ascii=False)

        #-----ROHAN'S CODE

        # face_count = 0
        #
        # # gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)


        # for (x, y, w, h) in faces:
        #
        #     # cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #     roi_gray = gray_image[y:y + h, x:x + w]
        #     roi_color = bgr_image[y:y + h, x:x + w]
        #     eyes = eye_cascade.detectMultiScale(roi_gray)
        #     face_count += 1
        #     text = "Person" + str(face_count)
        #     # cv.putText(img, text, (int(x + w/2), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #     atten(eyes, roi_color)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        print('except')
        continue
