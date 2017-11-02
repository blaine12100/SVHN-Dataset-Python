filename='extra.pickle'

with open(filename,'rb') as f:
    other=pickle.load(f)
    #print(other)


    train_data=other['train_dataset']
    test_data=other['test_dataset']
    #print(test_data)

    del other

train_dataset=train_data['X']
test_dataset=test_data['X']
#print(train_dataset)
#train_data = np.transpose(train_data, (3, 0, 1, 2))
train_labels=train_data['y']
test_labels=test_data['y']

print(len(test_dataset))
print(len(test_labels))
print(test_dataset.shape)
#print(train_dataset.length())

classes=10

batch_size=32

num_steps = 200000

graph=tf.Graph()

#Placeholder for the data
with graph.as_default():

    data_placeholder = tf.placeholder(tf.float32, shape=(batch_size,32,32,3))
    label_placeolder = tf.placeholder(tf.int64, shape=(batch_size, classes))

    tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size,32,32,3))

    tf_label_dataset = tf.placeholder(tf.float32, shape=(batch_size, classes))
    #Variable for Weights and Biases
    #1024 is for the number of nodes in each hidden layer

    layer1_weights=tf.Variable(tf.truncated_normal([3,3,3,16]))
    layer1_biases=tf.Variable(tf.zeros([16]))

    layer2_weights=tf.Variable(tf.truncated_normal([3,3,16,32]))
    layer2_biases=tf.Variable(tf.zeros([32]))

    layer3_weights=tf.Variable(tf.truncated_normal([2,2,32,64]))
    layer3_biases=tf.Variable(tf.zeros([64]))

    layer4_weights=tf.Variable(tf.truncated_normal([1024,10]))
    layer4_biases=tf.Variable(tf.zeros([10]))

    layer5_weights=tf.Variable(tf.truncated_normal([10,classes]))
    layer5_biases=tf.Variable(tf.zeros([classes]))

    #The last layer would be a fsoftmax layer with decay rate and l2 Regualrization

    def layer_multiplication(data_input_given,dropping=False):

        #Convolutional Layer 1

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

        print(Pool2.shape)

        CNN3 = tf.nn.relu(tf.nn.conv2d(Pool2, layer3_weights, strides=[1, 1, 1, 1], padding='SAME')) + layer3_biases
        print('CNN3 Done')
        #Third Pooling Layer

        Pool3 = tf.nn.max_pool(CNN3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print('Pool3 DOne')
        #Fully Connected Layer

        #print(Pool3.shape)

        shape = Pool3.get_shape().as_list()

        # print(shape)

        reshape = tf.reshape(Pool3, [shape[0], shape[1] * shape[2] * shape[3]])

        #print(reshape.shape)

        FullyCon = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)

        #print(FullyCon.shape)

        if dropping==False:
            print('Training')
            dropout = tf.nn.dropout(FullyCon, 0.6)
            z=tf.matmul(dropout,layer5_weights)+layer5_biases
            return z

        else:
            print('Testing')
            z = tf.matmul(FullyCon, layer5_weights) + layer5_biases
            return z


    gloabl_step = tf.Variable(0, trainable=False)

    decay_rate=tf.train.exponential_decay(1e-6,gloabl_step,4000,0.96,staircase=False,)

    train_input=layer_multiplication(data_placeholder,False)

    test_prediction = tf.nn.softmax(layer_multiplication(tf_test_dataset,True))

    #correct_prediction = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(label_placeolder, 1))

    #accu = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss=(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_placeolder,logits=train_input))
                                   + 0.01 * tf.nn.l2_loss(layer1_weights)
                                   + 0.01 * tf.nn.l2_loss(layer2_weights)
                                   + 0.01 * tf.nn.l2_loss(layer3_weights)
                                   + 0.01 * tf.nn.l2_loss(layer4_weights)
                                   + 0.01 * tf.nn.l2_loss(layer5_weights)
                                   )

    optimizer = tf.train.GradientDescentOptimizer(name='Stochastic', learning_rate=decay_rate).minimize(loss,global_step=gloabl_step)


    def accuracy(predictions, labels):
        print(predictions.shape[0])
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])

    config=tf.ConfigProto()
    config.gpu_options.allocator_type ='BFC'

    saver = tf.train.Saver()

    test_accuracy=[]

    with tf.Session(config=config) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        tf.train.write_graph(session.graph_def, '.', './SVHN.pbtxt')

        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            batch_test_data =  test_dataset[offset:(offset + batch_size), :, :]
            batch_test_labels = test_labels[offset:(offset + batch_size),:]
            #print(batch_data)
            #print(batch_test.shape)

            feed_dict = {data_placeholder:batch_data, label_placeolder:batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_input], feed_dict=feed_dict)
            if (step % 500 == 0):
                #print(session.run(decay_rate))
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                if(batch_test_data.shape!=(32,32,32,3)):
                    print('Skip')
                else:
                    correct_prediction = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(tf_label_dataset, 1))
                    accuracy_for_test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    print("Test Accuracy")

                    test_accuracy.append(accuracy_for_test.eval(feed_dict={tf_test_dataset:batch_test_data,
                                                tf_label_dataset:batch_test_labels}))

                    print(accuracy_for_test.eval(feed_dict={tf_test_dataset:batch_test_data,
                                                tf_label_dataset:batch_test_labels}))

        print(np.mean(test_accuracy))


        saver.save(sess=session, save_path='./SVHN.ckpt')
