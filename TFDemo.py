from data_process import cleanData
import numpy as np
import tensorflow as tf

cleanData = np.array(cleanData)

# spliting data
X_train = cleanData[:int(0.9 * len(cleanData)), :-1]
Y_train = cleanData[:int(0.9 * len(cleanData)), -1]

temp = []
for i in Y_train:
    temp.append([1,0]) if i == 2 else temp.append([0,1])

Y_train = np.array(temp)

X_test = cleanData[int(0.9 * len(cleanData)):, :-1]
Y_test = cleanData[int(0.9 * len(cleanData)):, -1]

temp = []
for i in Y_test:
    temp.append([1,0]) if i == 2 else temp.append([0,1])

Y_test = np.array(temp)

X = tf.placeholder(tf.float32, [None, 4], name="Input")

W1 = tf.Variable(tf.truncated_normal([4, 50], stddev=0.1), name="W1")
b1 = tf.Variable(tf.zeros([50]), name="B1")

W2 = tf.Variable(tf.truncated_normal([50, 2], stddev=0.1), name="W2")
b2 = tf.Variable(tf.zeros([2]), name="B2")

with tf.name_scope("First_Layer") as scope:
    l1 = tf.nn.leaky_relu(tf.matmul(X, W1) + b1)

tf.summary.histogram("weights1", W1)
tf.summary.histogram("biases1", b1)

with tf.name_scope("Second_Layer") as scope:
    l2 = tf.matmul(l1, W2) + b2

tf.summary.histogram("weights2", W2)
tf.summary.histogram("biases2", b2)

with tf.name_scope("Output") as scope:
    Y = tf.nn.softmax(l2)

Y_ = tf.placeholder(tf.float32, [None, 2], name="labels")

# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=l2,
#                                                         labels=Y_)
# cross_entropy = tf.reduce_mean(cross_entropy)*613

cross_entropy = -tf.reduce_sum(Y_*tf.log(Y))

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

tf.summary.scalar("cost_train", cross_entropy)
tf.summary.scalar("acc_train", accuracy)

learning_rate = 0.00003
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./log', graph=sess.graph)

    #500 epochs
    for i in range(500):
        acc, ce = sess.run([accuracy, cross_entropy], feed_dict={X: X_train,
                                                                 Y_: Y_train})
        if i%10 == 0:
            print("Accuracy train:", acc, "Train loss", ce)

        acc, ce = sess.run([accuracy, cross_entropy], feed_dict={X: X_test,
                                                                 Y_: Y_test})
        if i%10 == 0:
            print("Accuracy test:", acc, "Test loss", ce)

        _, summary = sess.run([optimizer, merged_summary_op], feed_dict={X: X_train, Y_: Y_train})
        writer.add_summary(summary, i * 613 + i)
