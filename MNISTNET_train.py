import numpy as np
import cv2
import os
import tensorflow as tf
import random as rd


num_classes, num_train, num_test = 14, 84000, 14000

# hyperparameters
display_step = 1
batch_size = 100
total_batch = 840

total_epoch = 400
learning_rate = 0.0005

cap = cv2.VideoCapture(0)
count = 0


def gen_ydata(num_classes, num_images):
    num_dw = num_images // num_classes
    y_data_t = []
    for n in range(num_images):
        arr_t = [0] * num_classes
        arr_t[n // num_dw] = 1
        y_data_t.append(arr_t)
    return np.array(y_data_t)

def shuffle_tr_te(x_data, y_data):
    num_images = len(x_data)
    numrd_arr = np.array(range(num_images))
    rd.shuffle(numrd_arr)

    _x_data, _y_data = np.zeros(x_data.shape), np.zeros(y_data.shape)
    for n in range(num_images): _x_data[n], _y_data[n] = x_data[numrd_arr[n]], y_data[numrd_arr[n]]
    return _x_data, _y_data

x_data, tx_data = np.load("train.npy"), np.load("test.npy")
y_data, ty_data = gen_ydata(14, num_train), gen_ydata(14, num_test)

x_data, y_data = shuffle_tr_te(x_data, y_data)

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
dropout_rate = tf.placeholder(tf.float32)

################################### Number Identifier(1st) ##########################################


# 2 x_data,y_data,W,b


X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
# img: ? x28 x28 x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 14])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.01))
#    Conv     -> (?, 28, 28, 64)
#    Pool     -> (?, 14, 14, 64)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=dropout_rate)

# L2 ImgIn shape=(?, 14, 14, 64)
W2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#    Conv      ->(?, 14, 14, 128)
#    Pool      ->(?, 7, 7, 128)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=dropout_rate)

# L3 ImgIn shape=(?, 7, 7, 128)
W3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
#    Conv      ->(?, 7, 7, 256)
#    Pool      ->(?, 4, 4, 256)
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=dropout_rate)

# L4 ImgIn shape=(?, 4, 4, 256)
W4 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01))
#    Conv      ->(?, 4, 4, 512)
#    Pool      ->(?, 2, 2, 512)
#    Reshape   ->(?, 2 * 2 * 512) # Flatten them for FC
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L4 = tf.nn.dropout(L4, keep_prob=dropout_rate)
L4_flat = tf.reshape(L4, [-1, 512 * 2 * 2])


# Final FC 2x2x512 inputs -> 10 outputs
def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


# L5 FC 512 * 2 * 2 Inputs -> 512 Outputs
W5 = tf.get_variable("W5", shape=[512 * 2 * 2, 512],
                     initializer=xavier_init(128 * 4 * 4, 512))
b1 = tf.Variable(tf.random_normal([512]))
h5 = tf.matmul(L4_flat, W5) + b1
_L5 = tf.nn.relu(h5)
L5 = tf.nn.dropout(_L5, dropout_rate)

# L6 FC 512 Inputs -> 256 Outputs
W6 = tf.get_variable("W6", shape=[512, 256], initializer=xavier_init(512, 256))
b2 = tf.Variable(tf.random_normal([256]))
h6 = tf.matmul(L5, W6) + b2
_L6 = tf.nn.relu(h6)
L6 = tf.nn.dropout(_L6, dropout_rate)

# L7 FC 256 Inputs -> 256 Outputs
W7 = tf.get_variable("W7", shape=[256, 256], initializer=xavier_init(256, 256))
b3 = tf.Variable(tf.random_normal([256]))
h7 = tf.matmul(L6, W7) + b3
_L7 = tf.nn.relu(h7)
L7 = tf.nn.dropout(_L7, dropout_rate)

# L8 FC 256 Inputs -> 14 Outputs
W8 = tf.get_variable("W8", shape=[256, 14], initializer=xavier_init(256, 14))
b4 = tf.Variable(tf.random_normal([14]))
h8 = tf.matmul(L7, W8) + b4

# 3 hypothesis,cost
hypothesis = h8
# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

# 4 AdamPropOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

predict_op = tf.argmax(hypothesis, 1)

################################Save System ##################################

ckpt_dir = "ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

global_step = tf.Variable(0, name='global_step', trainable=False)

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()

# This variable won't be stored, since it is declared after tf.train.Saver()
non_storable_variable = tf.Variable(777)

###############################################################################


# 5 Initialization
init = tf.global_variables_initializer()

# 6 Run
with tf.Session() as sess:
    sess.run(init)

    ################################Save System ##################################
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)  # restore all variables

    start = global_step.eval()  # get last global_step
    print("Start from:", start)
    ###############################################################################

    for epoch in range(start, total_epoch):

        avg_cost = 0.

        # Loop over all batches

        for i in range(total_batch):
            batch_xs = np.zeros((100, 28 * 28), dtype='float32')
            for j in range(100):
                batch_xs[j] = x_data[i * 100 + j]

            batch_ys = np.zeros((100, 14), dtype='float32')

            for j in range(100):
                batch_ys[j] = y_data[i * 100 + j]

                # Fit training using batch data

            sess.run(train, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: 1})

            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: 1}) / total_batch
            # show logs per epoch step

        global_step.assign(epoch).eval()  # set and update(eval) global_step with index, i
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)

        if epoch % display_step == 0:  # Softmax
            # Test model
            correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({X: tx_data, Y: ty_data, dropout_rate: 1}))
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")


