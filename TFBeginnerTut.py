import tensorflow as tf

#MNIST Data download:
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Data is 70k with a 50/10 split for train/test
#image size is 28x28 = 784
#onehot vectors presents the 10 levels as binary where the nth place represents n

'''
Softmax Regressions: Logistic regression model used for non-binary labels. Exponentiates
the weight of x, then normalizes it to a % out of 1. This gives different data significant changes
in weight. Look at http://neuralnetworksanddeeplearning.com/chap3.html#softmax
    Also read that entire site.

Equation used for this is:
    Output = softmax (Vector(X) __dot__ Vector(W) + b)
'''

#Variables

x = tf.placeholder(tf.float32, [None, 784])
    #Placeholder for input
    #None allows any value, and 784 flattened dimensional vector value of each image
W = tf.Variable(tf.zeros([784, 10]))
    #Variable is not input, but depends on input and will be addressed after running the session
    #Variable is a modifiable tensor
    #Sets variable tensor W to be full of 0's - in a 784 x 10 shape because there are 10 different labels hence 10 weights
b = tf.Variable(tf.zeros([10]))
    # Sets variable tensor b to be full of 0's, there are 10 because there are 10 types of images each with a bias.

#Model 
y = tf.nn.softmax(tf.matmul(x,W) + b)
    #softmax neural netowrk

'''
Training Data:
    Each model in ML has a cost, or loss - which is how far the model is from the ideal outcome.
    We try to minimize this.
    One way of determining cost is through something called 'Cross Entropy'
    Read More: http://colah.github.io/posts/2015-09-Visual-Information/
'''

#Cross Entropy
    #Create new variable for correct answers

y_ = tf.placeholder(tf.float32, [None, 10])
'''crossen = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    #Okay so - tf.log gets log of each y, then y_ is multiplied with that.
    #reduction_indices = [1] makes tf.reduce sum add values to second dimension of the tensor
    #tf.reduce_mean calculates the mean over all examples in the batch
'''
crossen = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
##crossen = tf.nn.(sparse_)softmax_cross_entropy_with_logits
    #recommended code becaused docstr version is unstable but understand the crossen equation

#Train Step w/ backpropagation alg:
    #http://colah.github.io/posts/2015-08-Backprop/

train = tf.train.GradientDescentOptimizer(0.5).minimize(crossen)
'''Optimization Algs : https://www.tensorflow.org/api_docs/python/train/#optimizers'''
    #Minimizes CrossEn with Gradient Descent alg at a learning rate of 0.5. 
    
#Initialize the variables:
init = tf.global_variables_initializer()

#Create Session and Launch Model
sess = tf.Session()
sess.run(init)

for i in range(1000):
    '''
Using batches to train is called Stochastic training.
Note that batch data replaces placeholders x and y_ below
    '''
    batch_xs, batch_ys = data.train.next_batch(100)
    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

#Prediction

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    #List of booleans to determine fraction correct.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x:data.test.images, y_: data.test.labels}))
