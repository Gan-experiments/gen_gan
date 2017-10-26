import tensorflow as tf
import numpy as np
import random
import datetime
from utils import*
import os 

batch_size=64
class batchnorm():
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,decay=self.momentum,updates_collections=None,epsilon=self.epsilon,scale=True,is_training=train,scope=self.name)
def lrelu(x, leak=0.2):
    return tf.maximum(x,x*leak)

def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
	x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv(x,num_filters,kernel=5,stride=[1,2,2,1],name="conv",padding='SAME'):
    with tf.variable_scope(name):
        w=tf.get_variable('w',shape=[kernel,kernel,x.get_shape().as_list()[3], num_filters],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        b=tf.get_variable('b',shape=[num_filters],
            initializer=tf.constant_initializer(0.0))
        con=tf.nn.conv2d(x, w, strides=stride, padding=padding)
        return tf.reshape(tf.nn.bias_add(con, b),con.shape)

def fcn(x,num_neurons,name="fcn"):#(without batchnorm )
    with tf.variable_scope(name):

        w=tf.get_variable('w',shape=[x.get_shape().as_list()[1],num_neurons],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        b=tf.get_variable('b',shape=[num_neurons],
            initializer=tf.constant_initializer(0.0))
        return tf.matmul(x,w)+b

def deconv(x,output_shape,kernel=5,stride=[1,2,2,1],name="deconv"):
    with tf.variable_scope(name):
        num_filters=output_shape[-1]
        w=tf.get_variable('w',shape=[kernel,kernel, num_filters,x.get_shape().as_list()[3]],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b=tf.get_variable('b',shape=[num_filters],
            initializer=tf.constant_initializer(0.0))
        decon=tf.nn.conv2d_transpose(x, w, strides=stride,output_shape=output_shape)
        return tf.reshape(tf.nn.bias_add(decon, b),decon.shape)

def generator(z,y):
    with tf.variable_scope("generator"):

        
	s_h, s_w = 28,28
	
	s_h2, s_h4 = int(s_h/2), int(s_h/4)

	s_w2, s_w4 = int(s_w/2), int(s_w/4)

	yb = tf.reshape(y, [batch_size, 1, 1,10])
	
        z_       =    concat([z,y],1)

        h0       =    fcn(z_,num_neurons=512*2,name="g_fcn0")

        gbn0     =    batchnorm(name="g_bn0")

	h0 = tf.nn.relu(gbn0(fcn(z,1024, 'g_h0_lin')))
        
	h0 = concat([h0, y], 1)
       

        gbn1     =    batchnorm(name="g_bn1")

	h1 = tf.nn.relu(gbn1(fcn(h0, 64*2*s_h4*s_w4, 'g_h1_lin')))

        gbn2     =    batchnorm(name="g_bn2")

	h1 = tf.reshape(h1, [batch_size, s_h4, s_w4, 64 * 2])
	
	h1 = conv_cond_concat(h1, yb)
	
	h2 = tf.nn.relu(gbn2(deconv(h1,[batch_size, s_h2, s_w2, 64 * 2], name='g_h2')))

	h2 = conv_cond_concat(h2, yb)	

       
        return tf.nn.sigmoid(deconv(h2, [batch_size, s_h, s_w, 1], name='g_h3'))

def sampler(z,y):
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()

        s_h, s_w = 28,28

        s_h2, s_h4 = int(s_h/2), int(s_h/4)

        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        yb = tf.reshape(y, [batch_size, 1, 1,10])

        z_	 =    concat([z,y],1)

        h0	 =    fcn(z_,num_neurons=512*2,name="g_fcn0")

        gbn0     =    batchnorm(name="g_bn0")

        h0 = tf.nn.relu(gbn0(fcn(z,1024, 'g_h0_lin'),train=False))

       	h0 = concat([h0, y], 1)


        gbn1     =    batchnorm(name="g_bn1")

        h1 = tf.nn.relu(gbn1(fcn(h0, 64*2*s_h4*s_w4, 'g_h1_lin'),train=False))

        gbn2     =    batchnorm(name="g_bn2")

        h1 = tf.reshape(h1, [batch_size, s_h4, s_w4, 64 * 2])

	h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(gbn2(deconv(h1,[batch_size, s_h2, s_w2, 64 * 2], name='g_h2'),train=False))

        h2 = conv_cond_concat(h2, yb)


        return tf.nn.sigmoid(deconv(h2, [batch_size, s_h, s_w, 1], name='g_h3'))

def discriminator(imgs,y,reuse=False):

    with tf.variable_scope("discriminator") as scope:

        if reuse:

            scope.reuse_variables()
        yb = tf.reshape(y, [batch_size, 1, 1, 10])
        x = conv_cond_concat(imgs, yb)        

        h0       =    lrelu(conv(x,11,name="d_h0"))

	h0 = conv_cond_concat(h0, yb)

	dbn1     =    batchnorm(name="d_bn1")

	h1 = lrelu(dbn1(conv(h0, 65, name='d_h1_conv')))

        h1 = tf.reshape(h1, [batch_size, -1])

	h1 = concat([h1, y], 1)

        dbn2     =    batchnorm(name="d_bn2")

	h2 = lrelu(dbn2(fcn(h1,1024, 'd_h2_lin')))
	
	h2 = concat([h2, y], 1)
	

	h3 = fcn(h2, 1, 'd_h3_lin')      

        return tf.nn.sigmoid(h3)

def load_mnist():
    data_dir = os.path.join("./data-1", "mnist")
   # data_dir="/home/satwik/Desktop/swaayatt_satwik/gan_test_Code/data /mnist"
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
	y_vec[i,y[i]] = 1.0
    
    return (X/255.),y_vec

images  =   tf.placeholder(tf.float32, [batch_size, 28,28,1], name='images')

labels      =   tf.placeholder(tf.float32, [batch_size, 10], name='labels')

z      =   tf.placeholder(tf.float32, [batch_size, 100], name='z')

X,Y    =   load_mnist()

with tf.device('/gpu:0'):

   # labels_one_hot  =   tf.one_hot(labels,10)
    
    gen_imgs        =   generator(z=z,y=labels)

    real_source     =   discriminator(images,labels,reuse=False)

    fake_source		=	discriminator(gen_imgs,labels,reuse=True)

    samples 			= 	sampler(z=z,y=labels)

    d_loss			=	tf.reduce_mean(tf.square(1-real_source+fake_source))

    g_loss 			=	tf.reduce_mean(tf.square(real_source-fake_source))#tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_source,labels= tf.ones_like(fake_source)))

    t_vars			=	tf.trainable_variables()

    d_vars			=	[var for var in t_vars if 'd_' in var.name]

    g_vars			=	[var for var in t_vars if 'g_' in var.name]

    d_opt			=	tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(d_loss,var_list=d_vars)

    g_opt  			= 	tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(g_loss,var_list=g_vars)

    init   = tf.global_variables_initializer()

    config = tf.ConfigProto()

    config.gpu_options.per_process_gpu_memory_fraction = 0.2

    with tf.Session(config=config) as sess:

    	sess.run(init)

    	for epoch in range(2000):

		batch_x = X[(epoch%batch_size)*batch_size :(epoch%batch_size+1)*batch_size ]

		batch_y = Y[(epoch%batch_size)*batch_size :(epoch%batch_size+1)*batch_size ]

		z_=np.random.uniform(-1, 1, [batch_size, 100]).astype(np.float32)
    		
		for n in range(2):

    			sess.run(d_opt,feed_dict={z:z_,images:batch_x.reshape([batch_size,28,28,1]),labels:batch_y})


    		sess.run(g_opt,feed_dict={z:z_,labels:batch_y,images:batch_x.reshape([batch_size,28,28,1])})

    		D_loss = sess.run(d_loss,feed_dict={z:z_,images:batch_x.reshape([batch_size,28,28,1]),labels:batch_y })

    		G_loss = sess.run(g_loss,feed_dict={z:z_,labels:batch_y,images:batch_x.reshape([batch_size,28,28,1]) })

    		print "d loss after epoch ",epoch," is ",D_loss

    		print "g loss after epoch ",epoch," is ",G_loss

    		if epoch % 10 ==0:
                
                	sample = sess.run(samples,feed_dict={z:z_,labels:batch_y})

               	 	save_images(sample, image_manifold_size(sample.shape[0]),
                          './{}/{:02d}.png'.format("dc_out", epoch))







