import pandas as pd
import numpy as np
def one_hot(vec,vals=26) :
    n=len(vec)
    out=np.zeros((n,vals))
    out[range(n),vec]=1
    return out


Y_train = pd.read_csv("sign_mnist_train.csv") 
Y_test = pd.read_csv("sign_mnist_test.csv") 

import csv
with open("sign_mnist_train.csv", "r") as f:
    reader = csv.reader(f)
    
X_train = np.array(Y_train.iloc[:,1:])
X_train = np.array([np.reshape(i, (28,28,1)) for i in X_train])/255
X_test = np.array(Y_test.iloc[:,1:])
X_test = np.array([np.reshape(i, (28,28,1)) for i in X_test])/255


train_label= one_hot(Y_train['label'].to_numpy(),26)
test_label= one_hot(Y_test['label'].to_numpy(),26)    

def next_batch(num, data=X_train, labels=train_label):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

import tensorflow as tf 

x=tf.placeholder(tf.float32,shape=[None,28,28,1])
y_true=tf.placeholder(tf.float32,shape=[None,26])
hold_prob = tf.placeholder(tf.float32)


def weights(shape):
    random=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(random)

def bias(shape):
    bias_vals=tf.constant(0.1,shape=shape)
    return tf.Variable(bias_vals)
def  conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def conv_layer(input_x,shape):
    W=weights(shape)
    b=bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W)+b)

def full_layer(input_layer,size):
    input_size=int(input_layer.get_shape()[1])
    W=weights([input_size,size])
    b=bias([size])
    return tf.matmul(input_layer,W)+b


conv_1=conv_layer(x,shape=[4,4,1,28])
conv_1_pooling=pool(conv_1)

conv_2=conv_layer(conv_1_pooling,shape=[4,4,28,64])
conv_2_pooling=pool(conv_2)

conv_flat=tf.reshape(conv_2_pooling,[-1,3136])

full_layer_one=tf.nn.relu(full_layer(conv_flat,784))

full_dropout=tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

y_pred=full_layer(full_dropout,26)

y_pred


cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
optimizer=tf.train.AdamOptimizer(learning_rate=0.003)
train=optimizer.minimize(cross_entropy)

init=tf.global_variables_initializer()

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    for i in range(323):
        
        batch=next_batch(85)
        sess.run(train,feed_dict={x:batch[0],y_true:batch[1],hold_prob:.5})
        if i%100==0:
            print('currently on {}'.format(i))
            print('accuracy :')
            matches=tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            acc=tf.reduce_mean(tf.cast(matches,tf.float32))
            
            
            print(sess.run(acc,feed_dict={x:X_test,y_true:test_label,hold_prob:1.0}))
            print('\n')