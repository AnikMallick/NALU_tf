import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EPOCHS = 100
BATCH_SIZE = 128
SUMMARY_DIRECTORY = #--SUMMARY_DIRECTORY
PRINT_PARAMETER = 5

def __variable_on_cpu(name, shape, initializer):   
    with tf.device('/cpu:0'):
        var = tf.get_variable(name = name, shape = shape, initializer = initializer,dtype = tf.float32)
    return var

def __add_summary_nalu(W_hat, M_hat, G, y):

    tf.summary.histogram('W_hat',W_hat)
    tf.summary.histogram('M_hat',M_hat)
    tf.summary.histogram('G',G)
    tf.summary.histogram('y',y)

def __add_summary_fc(weights, biases, layer, op = False):
        tf.summary.histogram('Weights',weights)
        tf.summary.histogram('Biases',biases)
        tf.summary.histogram('Activation',layer)
        if not op:
            tf.summary.scalar('Sparsity',tf.nn.zero_fraction(layer))

def NALU(inputs, output_dim, name = 'NALU', epsilon = 1e-8):
    with tf.name_scope(name):
        shape = (int(inputs.shape[-1]),output_dim)
        
        W_hat = __variable_on_cpu(name = name + '_W_hat', 
                                  shape = shape, 
                                  initializer = tf.contrib.layers.xavier_initializer())
        M_hat = __variable_on_cpu(name = name + '_M_hat', 
                                  shape = shape, 
                                  initializer = tf.contrib.layers.xavier_initializer())
        G = __variable_on_cpu(name = name + '_G', 
                                  shape = shape, 
                                  initializer = tf.contrib.layers.xavier_initializer())

        #NAC: a = Wx, W = tanh(W_hat) * σ(M_hat)
        
        W = tf.nn.tanh(W_hat) * tf.nn.sigmoid(M_hat)
        a = tf.matmul(inputs, W)
        
        #NALU: y = g*a + (1 - g)*m, m = expW(log(|x| + epsilon)), g = σ(Gx)
        
        m = tf.exp(tf.matmul(tf.log(tf.abs(inputs) + epsilon), W))
        g = tf.nn.sigmoid(tf.matmul(inputs, G))
        
        y = g*a + (1 - g)*m
        
        __add_summary_nalu(W_hat, M_hat, G, y)
                
        return y

def create_fc_layer(inputs,n_inputs,n_nodes,name = 'fc',use_relu = True):
    with tf.name_scope(name):
        weights = __variable_on_cpu(name = name + '_w',
                                    shape = [n_inputs,n_nodes], 
                                    initializer = tf.contrib.layers.xavier_initializer())
        biases = __variable_on_cpu(name = name + '_b',
                                   shape = [n_nodes],
                                   initializer=tf.constant_initializer(0.1))

        layer = tf.matmul(inputs,weights) + biases

        if use_relu:
            layer = tf.nn.relu(layer)
        op = False
        if name == "Output":
            op = True
        __add_summary_fc(weights, biases, layer,op = op)
    return layer

def build_with_fc(fc_node_list, X, y_true):
    INITIAL_LEARNING_RATE = 0.001
    
    fc_list = []
    for fc_layer,fc_layer_nodes in enumerate(fc_node_list):
        if fc_layer == 0:
            inputs = X
        else:
            inputs = fc_list[fc_layer - 1]
        if fc_layer == len(fc_node_list) - 1:
            use_relu = False
            name = 'Output'
        else:
            use_relu = True
            name = 'fc' + str(fc_layer)
        
        fc_list.append(create_fc_layer(inputs = inputs,
                                       n_inputs = inputs.shape[-1],
                                       n_nodes = fc_layer_nodes,
                                       name = name,
                                       use_relu = use_relu))
        
    onehot_labels_pred = tf.nn.softmax(fc_list[-1], name = 'pred_onehot_labels')
    y_pred = tf.argmax(onehot_labels_pred,axis = 1, name = 'pred_labels')
    with tf.name_scope('Loss'):  
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = fc_list[-1],
                                                                labels = y_true)
        loss = tf.reduce_mean(cross_entropy, name = 'model_cost')
    
    with tf.name_scope('Optimizer'):
        train_op = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(loss)
        
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_true,axis = 1),y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
    tf.summary.scalar('Loss',loss)
    tf.summary.scalar('Accuracy',accuracy)

    y_labels_true = tf.map_fn(list,[tf.argmax(y_true,axis = 1)])
        
    return train_op, loss, accuracy, y_pred, y_labels_true

def build_with_nalu(NALU_layer_size_list, X, y_true):
    INITIAL_LEARNING_RATE = 0.001
    nalu_list = []
    for layer,layer_size in enumerate(NALU_layer_size_list):
        if layer == 0:
            inputs = X
        else:
            inputs = nalu_list[layer - 1]
        type(inputs)
            
        nalu_list.append(NALU(inputs = inputs, 
                              output_dim = layer_size, 
                              name = 'NALU' + str(layer)))

    onehot_labels_pred = tf.nn.softmax(nalu_list[-1], name = 'pred_onehot_labels')
    y_pred = tf.argmax(onehot_labels_pred,axis = 1, name = 'pred_labels')
    with tf.name_scope('Loss'):  
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = nalu_list[-1],
                                                                labels = y_true)
        loss = tf.reduce_mean(cross_entropy, name = 'model_cost')
    
    with tf.name_scope('Optimizer'):
        train_op = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(loss)
        
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_true,axis = 1),y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
    tf.summary.scalar('Loss',loss)
    tf.summary.scalar('Accuracy',accuracy)

    y_labels_true = tf.map_fn(list,[tf.argmax(y_true,axis = 1)])
        
    return train_op, loss, accuracy, y_pred, y_labels_true

def f1_score(cm,name):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for cls in range(10):
        true_positive += cm[cls,cls]
        true_positive /= 6

        false_positive += (sum(cm[:,cls]) - cm[cls,cls])
        false_positive /= 6

        false_negative += (sum(cm[cls,:],1) - cm[cls,cls])
        false_negative /= 6

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * (precision * recall) / (precision + recall)

    print('for',name,'----precision:',precision,', recall:',recall,', f1:',f1)

def plot_confusion(confm,classes,title):
    plt.imshow(confm,cmap = 'gray')

    plt.colorbar()
    thr = confm.max()/2
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes)
    plt.yticks(tick_marks,classes)
    plt.title(title)
    plt.xlabel('Predicted_Labels')
    plt.ylabel('True_Labels')
    for i in range(confm.shape[0]):
        for j in range(confm.shape[1]):
            plt.text(j,i,confm[i,j],horizontalalignment="center",color="white" if confm[i, j] < thr else "black")
    plt.show()

mnist = input_data.read_data_sets('D:\\Python\\tensorflow_gpu_conda_env\\mnist\\dataset',one_hot= True)

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape = [None, 784])
y_true = tf.placeholder(tf.float32, shape = [None, 10])

fc_train_op, fc_loss, fc_accuracy, fc_y_pred, fc_y_labels_true = build_with_fc(fc_node_list = [50,20, 10], X = X, y_true = y_true)

msg = 'Epoch: {0},Completed out of: {1},loss in this epoch: {2}, Accuracy: {3}'

#run with FC
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    marged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(SUMMARY_DIRECTORY)
    writer.add_graph(sess.graph)
    n_batch = int(mnist.train.num_examples/BATCH_SIZE)

    #train
    for epoch in range(EPOCHS):
        #print(epoch)
        cost = 0
        acc = 0
        for _ in range(n_batch):
            batch_X, batch_y = mnist.train.next_batch(BATCH_SIZE)
            _, l,a = sess.run([fc_train_op, fc_loss,fc_accuracy], feed_dict = {X:batch_X, y_true:batch_y})
            cost += l
            acc += a
            if (epoch+1) % 5 == 0:
                s = sess.run(marged_summary, feed_dict = {X:batch_X, y_true:batch_y})
                writer.add_summary(s, epoch+1)
        if (epoch+1)%5 == 0:
            #a = sess.run(accuracy, feed_dict = {X:batch_X, y_true:batch_y})
            print(msg.format(epoch+1,EPOCHS,cost/n_batch,acc/n_batch))
        

    #test
    acc = 0
    pred = list()
    true = list()

    for _ in range(int(mnist.test.num_examples/BATCH_SIZE)):
        batch_X, batch_y = mnist.test.next_batch(BATCH_SIZE)
        a, p,t= sess.run([fc_accuracy,fc_y_pred,fc_y_labels_true], feed_dict = {X:batch_X, y_true:batch_y})
        acc += a
        '''
        pred.extend(list(p))
        true.extend(t)
        '''
        
    acc = acc/int(mnist.test.num_examples/BATCH_SIZE)
    print('Test_acc: ',acc)
    '''
    cm = confusion_matrix(y_true = true, y_pred = pred)
    plot_confusion(cm,list(set(true)),'FC_CONFM')
    f1_score(cm,'FC')
    '''

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape = [None, 784])
y_true = tf.placeholder(tf.float32, shape = [None, 10])

nalu_train_op, nalu_loss, nalu_accuracy, nalu_y_pred, nalu_y_labels_true = build_with_nalu(NALU_layer_size_list = [80,10],X = X, y_true = y_true)

#with nalu
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    marged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(SUMMARY_DIRECTORY)
    writer.add_graph(sess.graph)
    n_batch = int(mnist.train.num_examples/BATCH_SIZE)
    #train
    for epoch in range(EPOCHS):
        #print(epoch)
        cost = 0
        acc = 0
        for _ in range(n_batch):
            batch_X, batch_y = mnist.train.next_batch(BATCH_SIZE)
            _, l,a = sess.run([nalu_train_op, nalu_loss,nalu_accuracy], feed_dict = {X:batch_X, y_true:batch_y})
            cost += l
            acc += a
        if (epoch+1) % 5 == 0:
            s = sess.run(marged_summary, feed_dict = {X:batch_X, y_true:batch_y})
            writer.add_summary(s, epoch+1)
        if (epoch+1)%5 == 0:
            #a = sess.run(accuracy, feed_dict = {X:batch_X, y_true:batch_y})
            print(msg.format(epoch+1,EPOCHS,cost/n_batch,acc/n_batch))

    #test
    acc = 0
    pred = list()
    true = list()
    for _ in range(int(mnist.test.num_examples/BATCH_SIZE)):
        batch_X, batch_y = mnist.test.next_batch(BATCH_SIZE)
        a, p, t= sess.run([nalu_accuracy,nalu_y_pred,nalu_y_labels_true], feed_dict = {X:batch_X, y_true:batch_y})
        acc += a
        '''
        pred.extend(list(p))
        true.extend(t)
        '''
    acc = acc/int(mnist.test.num_examples/BATCH_SIZE)
    print('Test_acc: ',acc)
    '''
    cm = confusion_matrix(y_true = true, y_pred = pred)
    plot_confusion(cm,list(set(true)),'NALU_CONFM')
    f1_score(cm,'NALU')
    '''

