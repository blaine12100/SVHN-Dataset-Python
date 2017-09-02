import os
import sys
import tarfile
import numpy as np
import scipy.io as sio
from six.moves import cPickle as pickle
import tensorflow as tf
from PIL import Image
import datetime
import matplotlib.pyplot as plt
def OneHot(label,n_classes):
    label=np.array(label).reshape(-1)
    label=np.eye(n_classes)[label]

    return label

data_file = 'extra.pickle'

print('Tring to load pickle from %s' % data_file)
with open(data_file, 'rb') as file:
    svhn_datasets = pickle.load(file)
    train_dataset = svhn_datasets['train_dataset']
    test_dataset = svhn_datasets['test_dataset']
    del svhn_datasets # free up memory
print('pickle loaded successfully!')

train_data = train_dataset['X']
train_labels = train_dataset['y']

test_data = test_dataset['X']
test_labels = test_dataset['y']

del train_dataset, test_dataset

print('Test data:', test_data.shape,', Test labels:', test_labels.shape)

from sklearn.cross_validation import train_test_split

print ('Train data:', train_data.shape,', Train labels:', train_labels.shape)
#print ('Validation data:',,', Validation labels:', valid_labels.shape)

graph = tf.Graph()

with graph.as_default():
    # placeholders for input data batch_size x 32 x 32 x 3 and labels batch_size x 10
    data_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    label_placeholder = tf.placeholder(tf.float32, shape=[None, 10])

    # defining decaying learning rate
    global_step = tf.Variable(0)
    decay_rate = tf.train.exponential_decay(1e-4, global_step=global_step, decay_steps=10000, decay_rate=0.97)

    layer1_weights = tf.Variable(tf.truncated_normal([3, 3, 3, 64],stddev=0.1))
    layer1_biases = tf.Variable(tf.constant(0.1, shape=[64]))

    layer2_weights = tf.Variable(tf.truncated_normal([3, 3, 64,32],stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(0.1,shape=[32]))

    layer3_weights = tf.Variable(tf.truncated_normal([2, 2, 32, 4096],stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(0.1,shape=[4096]))

    layer4_weights = tf.Variable(tf.truncated_normal([4096,10],stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(0.1,shape=[10]))

    layer5_weights = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
    layer5_biases = tf.Variable(tf.constant(0.1, shape=[10]))



    def layer_multiplication(data_input_given):

        #Convolutional Layer 1

        data_input_given=np.reshape(data_input_given,[-1,32,32,3])

        CNN1=tf.nn.relu(tf.nn.conv2d(data_input_given,layer1_weights,strides=[1,1,1,1],padding='SAME')+layer1_biases)

        print('CNN1 Done!!')

        #Pooling Layer

        Pool1=tf.nn.max_pool(CNN1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        print('Pool1 DOne')

        #second Convolution layer

        CNN2=tf.nn.relu(tf.nn.conv2d(Pool1,layer2_weights,strides=[1,1,1,1],padding='SAME'))+layer2_biases
        print('CNN2 Done')
        #Second Pooling

        Pool2 = tf.nn.max_pool(CNN2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print('pool2 Done')
        #Third Convolutional Layer

        CNN3 = tf.nn.relu(tf.nn.conv2d(Pool2, layer3_weights, strides=[1, 1, 1, 1], padding='SAME')) + layer3_biases
        print('CNN3 Done')
        #Third Pooling Layer

        Pool3 = tf.nn.max_pool(CNN3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print('Pool3 DOne')
        #Fully Connected Layer

        FullyCon=tf.reshape(Pool3,[-1,4096])

        FullyCon=tf.nn.relu(tf.matmul(FullyCon,layer4_weights)+layer4_biases)

        print('Fullyconnected Done')
        dropout = tf.nn.dropout(FullyCon, 0.4)

        dropout=tf.reshape(dropout,[-1,1000])

        dropout=tf.matmul(dropout,layer5_weights)+layer5_biases

        print(dropout.shape)


        return dropout


    train_input = layer_multiplication(train_data)
    print(train_input.shape)

    loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_placeholder, logits=train_input))
            + 0.01 * tf.nn.l2_loss(layer1_weights)
            + 0.01 * tf.nn.l2_loss(layer2_weights)
            + 0.01 * tf.nn.l2_loss(layer3_weights)
            + 0.01 * tf.nn.l2_loss(layer4_weights)
            )

    optimizer = tf.train.GradientDescentOptimizer(name='Stochastic', learning_rate=decay_rate).minimize(loss,global_step=global_step)

    #print(train_input.shape)


    batch_size = 64

    num_steps=100000

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        for i in range(num_steps):
            print("in loop")
            offset = (i * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_data[offset:(offset + batch_size), :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            print("training")
            feed_dict = {data_placeholder: batch_data, label_placeholder: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_input], feed_dict=feed_dict)
            print('Finished')

