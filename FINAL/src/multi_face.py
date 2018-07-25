import numpy as np
import cv2 as cv
import numpy as np
import tensorflow as tf
import sys
import glob
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

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
SAVE_PATH = "./checkpoint/ver1.0_iteration.64000.ckpt"
saver.restore(sess, SAVE_PATH)

prev_face = [(0,0,30,30)]
prev_eyes = [(1,1,1,1), (1,1,1,1)]
drowsiness_check_list = [0] * WINDOW_SIZE
drowsiness_check_idx = 0

def atten(eyes,roi_color):
	eye_count = 1
	global drowsiness_check_idx
	for (ex,ey,ew,eh) in eyes:
		cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		eye_image = roi_color[ey:ey+eh , ex:ex+ew]
		input_images = cv.resize(eye_image, (32,32))
		input_images.resize((1,32,32,3))
		input_images = np.divide(input_images, 255.0)
		label = sess.run(tf.argmax(y_conv, 1), feed_dict={keep_prob:1.0, x_tensor:input_images})
		drowsiness_check_list[drowsiness_check_idx%WINDOW_SIZE] = label[0]
		drowsiness_check_idx += 1
		if eye_count == 2:
			if drowsiness_check_list == [1] * WINDOW_SIZE:
				print("Face - ",face_count," - Not Attentive",)
			elif drowsiness_check_list == [0] * WINDOW_SIZE:
				print("Face - ",face_count," - Attentive")
		eye_count+=1
		cv.imshow("Face",img)





cap = cv.VideoCapture(0)
while True:
	img = cap.read()[1]
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	face_count = 0
	for (x,y,w,h) in faces:
		cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		face_count += 1
		text = "Person" + str(face_count)
		cv.putText(img, text, (int(x + w/2), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		atten(eyes,roi_color)
	if cv.waitKey(1) & 0xFF == ord('q'):break

cap.release()
cv.destroyAllWindows()
