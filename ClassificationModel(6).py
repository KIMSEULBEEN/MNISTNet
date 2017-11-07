#1 import
import tensorflow as tf
import numpy as np
import cv2
import os
import random
import sys
from PIL import Image

cap = cv2.VideoCapture(0)
count = 0

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
dropout_rate = tf.placeholder(tf.float32)

y_out = 16
################################### Number Identifier(1st) ##########################################


#2 x_data,y_data,W,b

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   
                 # img: ? x28 x28 x1 (black/white)
Y = tf.placeholder(tf.float32, [None, y_out])


# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.01))
#    Conv     -> (?, 28, 28, 64)
#    Pool     -> (?, 14, 14, 64)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=dropout_rate)


# L2 ImgIn shape=(?, 14, 14, 64)
W2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#    Conv      ->(?, 14, 14, 128)
#    Pool      ->(?, 7, 7, 128)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=dropout_rate)


# L3 ImgIn shape=(?, 7, 7, 128)
W3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
#    Conv      ->(?, 7, 7, 256)
#    Pool      ->(?, 4, 4, 256)
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=dropout_rate)


# L4 ImgIn shape=(?, 4, 4, 256)
W4 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01))
#    Conv      ->(?, 4, 4, 512)
#    Pool      ->(?, 2, 2, 512)
#    Reshape   ->(?, 2 * 2 * 512) # Flatten them for FC
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L4 = tf.nn.dropout(L4, keep_prob=dropout_rate)
L4_flat = tf.reshape(L4, [-1, 512 * 2 * 2])


# Final FC 512*2*2 inputs -> 16 Outputs
def xavier_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs) )
        return tf.random_uniform_initializer(-init_range,init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev = stddev)


# L5 FC 512 * 2 * 2 Inputs -> 512 Outputs
W5 = tf.get_variable("W5", shape=[512*2*2, 512], initializer=xavier_init(128*4*4,512))
b1 = tf.Variable(tf.random_normal([512]))
h5 = tf.matmul(L4_flat,W5) + b1
_L5 = tf.nn.relu(h5)
L5 = tf.nn.dropout(_L5,dropout_rate)


# L6 FC 512 Inputs -> 256 Outputs
W6 = tf.get_variable("W6", shape=[512, 256], initializer=xavier_init(512,256))
b2 = tf.Variable(tf.random_normal([256]))

h6 = tf.matmul(L5,W6) + b2
_L6 = tf.nn.relu(h6)
L6 = tf.nn.dropout(_L6,dropout_rate)


# L7 FC 256 Inputs -> 256 Outputs
W7 = tf.get_variable("W7", shape=[256, 256], initializer=xavier_init(256,256))
b3 = tf.Variable(tf.random_normal([256]))

h7 = tf.matmul(L6,W7) + b3
_L7 = tf.nn.relu(h7)
L7 = tf.nn.dropout(_L7,dropout_rate)

# L8 FC 256 Inputs -> 16 Outputs
W8 = tf.get_variable("W8", shape=[256, y_out], initializer=xavier_init(256,14))
b4 = tf.Variable(tf.random_normal([14]))
h8 = tf.matmul(L7,W8) + b4



#3 hypothesis,cost
hypothesis = h8
# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))



#4 AdamPropOptimizer
learning_rate = 0.0005
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(cost)

predict_op = tf.argmax(hypothesis, 1)

################################Save System ##################################

ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

global_step = tf.Variable(0, name='global_step', trainable=False)

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()

# This variable won't be stored, since it is declared after tf.train.Saver()
non_storable_variable = tf.Variable(777)

###############################################################################



#5 Initialization
init = tf.global_variables_initializer()

################################# 디렉토리 생성 ###########################
dir_name = "original_trim"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

dir_name = "resize"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
################################# 디렉토리 생성 ###########################

limlabelnum = 30

################################### 이전 resize 파일 제거 ###############################

for label in range(limlabelnum):
    ex = os.path.isfile("resize//resize"+str(label)+".jpg")
    if(str(ex) == "True"):
        os.remove("resize//resize"+str(label)+".jpg")

for label in range(limlabelnum):
    ex = os.path.isfile("original_trim//original_trim"+str(label)+".jpg")
    if(str(ex) == "True"):
        os.remove("original_trim//original_trim"+str(label)+".jpg")

################################### 이전 resize 파일 제거 ###############################


########################################### Number Identifier(1st) #############################################

while(True):
    sys.stdout.flush()

    count = count + 1
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    
    ################################### Image Preprocessing ############################################
    if count % 250 == 0:
        cv2.imwrite("original.jpg", frame)

        img = cv2.imread("original.jpg", 1)
        img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurcolor= cv2.GaussianBlur(img, (5,5),0)
        blurgray= cv2.GaussianBlur(img_gray,(5,5),0)

        adaptive_th= cv2.adaptiveThreshold(blurgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)
        adaptive_th_inv= cv2.adaptiveThreshold(blurgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)

        cv2.imwrite("original.jpg", frame)
        cv2.imwrite("threshdouble.jpg",adaptive_th)
        ################################### 이전 resize 파일 제거 ###############################
        if(count > 250):
            for label in range(labelnum):
                ex = os.path.isfile("resize//resize"+str(label)+".jpg")
                if(str(ex) == "True"):
                    os.remove("resize//resize"+str(label)+".jpg")

            for label in range(labelnum):
                ex = os.path.isfile("original_trim//original_trim"+str(label)+".jpg")
                if(str(ex) == "True"):
                    os.remove("original_trim//original_trim"+str(label)+".jpg")
       ################################### 이전 resize 파일 제거 ###############################    
        labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(adaptive_th_inv)
        if(labelnum > limlabelnum):
            labelnum = limlabelnum

        for i in range(0, labelnum-1):
            for j in range(1, labelnum-i-1):
                if(contours[j][0]>contours[j+1][0]):
                   for k in range(0, 4):
                       temp=contours[j][k]
                       contours[j][k]=contours[j+1][k]
                       contours[j+1][k]=temp
                j+=1
            i+=1                       

        for label in range(1,labelnum):

            x,y,w,h,size = contours[label] 

            if w>30 or h > 30:
                
                img = cv2.rectangle(adaptive_th, (x,y), (x+w,y+h), (255,255,255), 1) 
                img_trim = img[y:y+h, x:x+w]
                
                cv2.imwrite("original_trim//original_trim"+str(label)+".jpg",img_trim)

                img = cv2.imread("original_trim//original_trim"+str(label)+".jpg",0)
                im = Image.open("original_trim//original_trim" +str(label)+".jpg")

                if w/h > 2.5:
                     img_ = np.zeros((int(5.5*h),int(1.6*w)),dtype = 'float32')
                     for i in range(int(h)):
                          for j in range(int(w)):
                             img_[int(2.2*h+i),int(0.3*w+j)] = 255. - im.getpixel((j,i))
                elif h/w > 2.5:
                     img_ = np.zeros((int(1.6*h),int(5.5*w)),dtype = 'float32')
                     for i in range(int(h)):
                          for j in range(int(w)):
                             img_[int(0.3*h+i),int(2.2*w+j)] = 255. - im.getpixel((j,i))
                else:
                     img_ = np.zeros((int(1.6*h),int(1.6*w)),dtype = 'float32')
                     for i in range(int(h)):
                         for j in range(int(w)):
                           img_[int(0.3*h+i),int(0.3*w+j)] = 255. - im.getpixel((j,i))


                cv2.imwrite("resize//resize" + str(label) + ".jpg",img_)
                    
                img = cv2.imread('resize//resize' + str(label) + '.jpg',0)
                res = cv2.resize(img,(28, 28), interpolation = cv2.INTER_CUBIC)
                cv2.imwrite("resize//resize" + str(label) + ".jpg",res)

    ################################### Image Preprocessing ############################################

    ########################################### Number Identifier(2nd) #############################################

        #6 Run
        with tf.Session() as sess:
            sess.run(init)

            ############################## Save System(2nd) ####################################
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                          
                saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

                start = global_step.eval() # get last global_step
                ############################## Save System(2nd) ####################################

                a = 0
                b = 0
                ifp = 0
                ifm = 0
                ifm2 = 0
                ifd = 0              

                ################################## Real Image Test ################################
                img_ = np.zeros((28,28),dtype = 'float32')
                img = np.ones((1,784),dtype = 'float32')

                numbers = ""  # Merged Number Variable
                for n in range(1,labelnum):
                    ex = os.path.isfile("resize//resize"+str(n)+".jpg")
                    if(str(ex) == "True"):                         
                        im = Image.open("resize//resize" + str(n) + ".jpg")

                        for i in range(28):
                            for j in range(28):
                                img_[i,j] = im.getpixel((i,j)) /255.
                                
                    
                        for i in range(28):
                            for j in range(28):
                                img[0,28*j + i] = img_[i,j]
                        num = sess.run(predict_op,feed_dict={X: img,dropout_rate: 1} )

                        if int(num) == 10:
                            num = "[+]"

                        elif int(num) == 11:
                            num = "[-]"

                        elif int(num) == 12:
                            num = "[*]"
                        elif int(num) == 13:
                            num = "[/]"

                        elif int(num) == 14:
                            num = '[A]"

                        elif int(num) == 15:
                            num = "[~]"
                                                   
                        numbers += str(num) + " "
                        if ifp == 0 and ifm == 0 and ifm2 == 0 and ifd == 0:
                            if str(num) == "[+]":
                                ifp += 1 

                            elif str(num) == "[-]":
                                ifm += 1

                            elif str(num) == "[*]":
                                ifm2 += 1 

                            elif str(num) == "[/]":
                                ifd += 1 

                            else:
                                a *= 10
                                a += int(num)
                        else:
                            if str(num) != "[+]" and str(num) != "[-]" and str(num) != "[*]" and str(num) != "[/]":
                                b *= 10
                                b += int(num) 
                    if n == labelnum-1 and ifp == 1:
                        print("Equation: %d + %d = " % (int(a),int(b)) + str(a+b))
                           
                    if n == labelnum-1 and ifm == 1:
                        print("Equation: %d - %d = " % (int(a),int(b)) + str(a-b))    

                    if n == labelnum-1 and ifm2 == 1:
                        print("Equation: %d * %d = " % (int(a),int(b)) + str(a*b))  

                    if n == labelnum-1 and ifd == 1 and b != 0:
                        print("Equation: %d / %d = " % (int(a),int(b)) + str(a/b))  
                ################################## Real Image Test ################################
        print("Prediction: " + numbers )
        print()
                  
    ########################################### Number Identifier(2nd) #############################################

    ## End Program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
