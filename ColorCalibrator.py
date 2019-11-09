import tensorflow as tf
import cv2
print("bok")


import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


img = cv2.imread('./processed_small2.jpg',cv2.IMREAD_COLOR)
new_img = cv2.imread('./processed_small2.jpg',cv2.IMREAD_COLOR)
h = img.shape[0]
w = img.shape[1]

'''
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
num_input = 3
num_classes = 3

n_hidden_1 = 7
n_hidden_2 = 10
n_hidden_3 = 10
n_hidden_4 = 10
n_hidden_5 = 7
learning_rate = 0.0005
num_steps = 50000

X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
L = tf.placeholder("float", None)

#training_input = [, [196.265 ,38.9964, 210.759], [169.048, 242.327, 221.563] ]
#training_output = [, [111.044, 51.0147, 202.492], [80.2594, 228.866, 211.277] ]
training_input = [[196.337, 165.561, 113.95], [196.0, 38.0, 210.0], [169.048, 242.327, 221.563], [245.23, 230.8, 227.2], [160, 210, 85]]
training_output = [[112.792 ,155.086, 131.88],[111.0, 51.0, 202.0], [80.2594, 228.866, 211.277], [204.1, 211.5, 214.1], [75, 193, 108]]
testing_input = []
for y in range(0, h):
    for x in range(0, w):
        testing_input += [img[y, x]]

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    out_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
    return out_layer


logits = neural_net(X)
formula = tf.abs(Y-logits)
f_output = tf.add(formula, 0)
y_output = tf.add(Y, 0)
cost = tf.reduce_mean(tf.reduce_sum(formula, reduction_indices=1))
cost_c = tf.add(tf.multiply(cost, 100),0)
tf.summary.scalar("chapter_test16", cost_c)
optimizer = tf.train.AdamOptimizer(learning_rate=L)
training_op = optimizer.minimize(cost)
init = tf.global_variables_initializer()
merge = tf.summary.merge_all()
saver = tf.train.Saver()


def train():
    global learning_rate, num_steps, training_input, training_output, num_input, num_classes, X, Y, weights, biases


    with tf.Session() as sess:

        print("learning")
        print(training_input)
        print(training_output)
        sess.run(init)
        train_writer = tf.summary.FileWriter('./logs/4/training ', sess.graph)
        for step in range(0, num_steps):
            y, f, l, _, c, summary = sess.run([y_output, f_output, logits, training_op, cost_c, merge], feed_dict={X: training_input, Y: training_output, L:learning_rate})
            if c < 3000:
                learning_rate = 0.00004
            elif c<100:
                learning_rate = 0.000005
            elif c<50:
                learning_rate = 0.0000002
            elif c < 10:
                    learning_rate = 0.000001
            train_writer.add_summary(summary, step)
            print("step=" , step, " cost= {:.9f}".format(c))


        save_path = saver.save(sess, "./chapter_test11_5clrs.ckpt")

        print("Model saved in path: %s" % save_path)


def test():
    global testing_input, new_img, img
    h = img.shape[0]
    w = img.shape[1]

    with tf.Session() as sess:

        saver.restore(sess, "./chapter_test11_5clrs.ckpt")

        for y in range(0, w):
            for x in range(0, h):
                #print(img[x,y])
                #print("-")
                #print(sess.run(logits, feed_dict={X: [img[x,y]]})[0])
                new_img[x, y] = sess.run(logits, feed_dict={X: [img[x,y]]})[0]
        print(sess.run(logits, feed_dict={X: training_input}))
        print("ok ")

        cv2.imshow('image', new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("testing_input ... ")
        print("-")

        cv2.imwrite("./image_fixed_5_colors.jpg", new_img)
        print("spremio")
        #run_full = sess.run(logits, feed_dict={X: testing_input})




train()
test()

