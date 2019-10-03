import tensorflow as tf
import os
import glob
import numpy as np
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt

def conv2d(input_map,number_of_outputs,num,name,stride,kernel_zise=3):
	#name = 'Conv2d_'+str(num)
	weights = tf.get_variable(name='cw_'+str(num),shape=[kernel_zise, kernel_zise, input_map.get_shape()[-1], number_of_outputs],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
	biases = tf.get_variable(name='cb_'+str(num),shape=[number_of_outputs],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
	conv = tf.nn.conv2d(input = input_map, filter = weights , strides= [1,stride,stride,1],padding='SAME',name = name)
	conv = tf.nn.bias_add(conv,biases)
	conv = tf.keras.layers.BatchNormalization()(conv)
	return conv

def convTranspose2d(input_map,output_shape,num,kernel_zise=2):
	name = 'deconv2d_'+str(num)
	weights = tf.get_variable(name='dw_'+str(num),shape=[kernel_zise, kernel_zise, output_shape[-1], input_map.get_shape()[-1]],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
	biases = tf.get_variable(name='db_'+str(num),shape=[output_shape[-1]],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
	deconv = tf.nn.conv2d_transpose(input_map,filter = weights, output_shape = output_shape , strides=[1, 2, 2, 1], padding='SAME', name = name)
	deconv = tf.nn.bias_add(deconv,biases)
	deconv = tf.keras.layers.BatchNormalization()(deconv)
	return deconv

def maxPool(input_map,name):
	return tf.nn.max_pool(value = input_map, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],padding='SAME', name = name)



def concat_conv_deconv( source, target):
	#print(source.shape)
	#print(target.shape)
	return tf.concat(axis =3,values=[source, target])
 



def Network(input_image,batch_size):

	num_of_encoder_channels = 32
	input_image_shape = input_image.shape
	
	#current = input_image

	############################Conv#########################################################################################
	conv1_1 = conv2d(input_map = input_image,num = 1,name = 'conv1_1', number_of_outputs = num_of_encoder_channels * (2 ** 1),stride = 2)
	conv1_1 = tf.nn.relu(conv1_1)
	print(conv1_1.shape)
	#conv1_2 = conv2d(input_map = conv1_1,num = 2,name = 'conv1_2', number_of_outputs = num_of_encoder_channels * (2 ** 1))
	#conv1_2 = tf.nn.relu(conv1_2)
	#print(conv1_2.shape)
	#conv2_1 = maxPool(conv1_2,name = 'maxpool1')
	#print(conv2_1.shape)
	conv2_2 = conv2d(input_map = conv1_1,num = 3,name = 'conv2_2', number_of_outputs = num_of_encoder_channels * (2 ** 2),stride = 2)
	conv2_2 = tf.nn.relu(conv2_2) 
	print(conv2_2.shape)
	#conv2_3 = conv2d(input_map = conv2_2,num = 4,name = 'conv2_3', number_of_outputs = num_of_encoder_channels * (2 ** 2))
	#conv2_3 = tf.nn.relu(conv2_3) 
	#conv3_1 = maxPool(conv2_3,name = 'maxpool2')
	conv3_2 = conv2d(input_map = conv2_2,num = 18,name = 'conv3_2', number_of_outputs = num_of_encoder_channels * (2 ** 3),stride = 2)
	conv3_2 = tf.nn.relu(conv3_2) 
	print(conv3_2.shape)
	#conv3_3 = conv2d(input_map = conv3_2,num = 19,name = 'conv3_3', number_of_outputs = num_of_encoder_channels * (2 ** 3))
	#conv3_3 = tf.nn.relu(conv3_3) 
	#conv4_1 = maxPool(conv3_3,name = 'maxpool3')
	#print(conv4_1.shape)
	conv4_2 = conv2d(input_map = conv3_2,num = 5,name = 'conv4_2', number_of_outputs = num_of_encoder_channels * (2 ** 4),stride = 2)
	conv4_2 = tf.nn.relu(conv4_2) 
	print(conv4_2.shape)
	#conv4_3 = conv2d(input_map = conv4_2,num = 6,name = 'conv4_3', number_of_outputs = num_of_encoder_channels * (2 ** 4))
	#conv4_3 = tf.nn.relu(conv4_3) 
	#conv5_1 = maxPool(conv4_3,name = 'maxpool4')
	conv5_2 = conv2d(input_map = conv4_2,num = 20,name = 'conv5_2', number_of_outputs = num_of_encoder_channels * (2 ** 5),stride = 2)
	conv5_2 = tf.nn.relu(conv5_2) 
	print(conv5_2.shape)
	#conv5_3 = conv2d(input_map = conv5_2,num = 7,name = 'conv5_3', number_of_outputs = num_of_encoder_channels * (2 ** 5))
	#conv5_3 = tf.nn.relu(conv5_3) 

########################################################################################################################################
	
	decon4_1  = convTranspose2d(input_map = conv5_2,output_shape= [batch_size,45,80,num_of_encoder_channels * (2 ** 4)],num=8)
	print(decon4_1.shape)
	decon4_1 = concat_conv_deconv(conv4_2,decon4_1)
	print(decon4_1.shape)
	decon4_2 = conv2d(input_map = decon4_1,num = 9,name = 'conv1_9', number_of_outputs = num_of_encoder_channels * (2 ** 4),stride = 1)
	decon4_2 = tf.nn.relu(decon4_2)
	print(decon4_2.shape)
	#decon4_3 = conv2d(input_map = decon4_2,num = 10,name = 'conv1_10', number_of_outputs = num_of_encoder_channels * (2 ** 4))
	#decon4_3 = tf.nn.relu(decon4_3)


	decon3_1  = convTranspose2d(input_map = decon4_2,output_shape= [batch_size,90,160,num_of_encoder_channels * (2 ** 3)],num=11)
	print(decon3_1.shape)
	decon3_1 = concat_conv_deconv(conv3_2,decon3_1)
	print(decon3_1.shape)
	decon3_2 = conv2d(input_map = decon3_1,num = 11,name = 'conv1_11', number_of_outputs = num_of_encoder_channels * (2 ** 3),stride=1)
	decon3_2 = tf.nn.relu(decon3_2)
	print(decon3_2.shape)
	# decon3_3 = conv2d(input_map = decon3_2,num = 12,name = 'conv1_12', number_of_outputs = num_of_encoder_channels * (2 ** 3))
	# decon3_3 = tf.nn.relu(decon3_3)


	decon2_1  = convTranspose2d(input_map = decon3_2,output_shape= [batch_size,180,320,num_of_encoder_channels * (2 ** 2)],num=13)
	print(decon2_1.shape)
	decon2_1 = concat_conv_deconv(conv2_2,decon2_1)
	print(decon2_1.shape)
	decon2_2 = conv2d(input_map = decon2_1,num = 13,name = 'conv1_13', number_of_outputs = num_of_encoder_channels * (2 ** 2),stride=1)
	decon2_2 = tf.nn.relu(decon2_2)
	print(decon2_2.shape)
	# decon2_3 = conv2d(input_map = decon2_2,num = 14,name = 'conv1_14', number_of_outputs = num_of_encoder_channels * (2 ** 2))
	# decon2_3 = tf.nn.relu(decon2_3)


	decon1_1  = convTranspose2d(input_map = decon2_2,output_shape= [batch_size,360,640,num_of_encoder_channels * (2 ** 1)],num=15)
	print(decon1_1.shape)
	decon1_1 = concat_conv_deconv(conv1_1,decon1_1)
	print(decon1_1.shape)	
	decon1_2 = conv2d(input_map = decon1_1,num = 15,name = 'conv1_15', number_of_outputs = num_of_encoder_channels * (2 ** 1),stride=1)
	decon1_2 = tf.nn.relu(decon1_2)
	# decon1_3 = conv2d(input_map = decon1_2,num = 16,name = 'conv1_16', number_of_outputs = num_of_encoder_channels * (2 ** 1))
	# decon1_3 = tf.nn.relu(decon1_3)

	############################################################################################################################################



	decon1_0 = conv2d(input_map = decon1_2,num = 17,name = 'conv1_17', number_of_outputs = 1,stride=1)
	decon1_0 = tf.nn.sigmoid(decon1_0)
	print(decon1_0.shape)

	return decon1_0

def data():
	train_data_path = "train_data\\"
	train_mask_path = "train_label\\"
	val_data_path = "val_data\\"
	val_mask_path = "val_label\\"

	data_path = os.path.join(train_data_path,'*jpg')
	files = glob.glob(data_path)
	#print(files)

	data_path1 = os.path.join(train_mask_path,'*jpg')
	files1 = glob.glob(data_path1)
	#print(files)

	data_path2 = os.path.join(val_data_path,'*jpg')
	files2 = glob.glob(data_path2)
	#print(files)
	data_path3 = os.path.join(val_mask_path,'*jpg')
	files3 = glob.glob(data_path3)
	#print(files)




	train_data = files[0:7000]
	train_labels = files1[0:7000]
	val_data = files2[0:2000]
	val_labels = files3[0:2000]

	return train_data,train_labels,val_data,val_labels



if __name__ == '__main__':

	batch_size = 8
	num_epochs = 100
	os.makedirs('Output_files')
	output_dir = "Output_files/"
	train_loss_list = []
	val_loss_list = []
	sess = tf.Session()

	with sess.as_default():
		input_image = tf.placeholder(tf.float32,[batch_size,720,1280,1],name="input_image")

		output_mask = tf.placeholder(tf.float32,[batch_size,360,640,1],name="output_mask")

		train_data,train_labels,val_data,val_labels = data()

		X_train = np.array(train_data)
		y_train = np.array(train_labels)
		X_val = np.array(val_data)
		y_val = np.array(val_labels)

		xtrain_dataset = tf.data.Dataset.from_tensor_slices(np.array(X_train))
		ytrain_dataset = tf.data.Dataset.from_tensor_slices(np.array(y_train))
		xval_dataset = tf.data.Dataset.from_tensor_slices(np.array(X_val))
		yval_dataset = tf.data.Dataset.from_tensor_slices(np.array(y_val))

		number_of_batches = int(len(train_data)/batch_size)

		print(number_of_batches)

		pred = Network(input_image,batch_size)

		train_loss = tf.losses.sigmoid_cross_entropy(output_mask, pred)
		train_loss_OP = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(train_loss)

		init = tf.global_variables_initializer()
		init_l = tf.local_variables_initializer()
		sess.run(init)
		sess.run(init_l)
		saver = tf.train.Saver()

		for epochs in range(1):

				combindedTrainDataset = tf.data.Dataset.zip((xtrain_dataset, ytrain_dataset)).shuffle(X_train.shape[0]).batch(batch_size)
				iterator = combindedTrainDataset.make_initializable_iterator()
				next_element = iterator.get_next()
				sess.run(iterator.initializer)
				for batches in range(2):
					image_batch = []
					val = sess.run(next_element)
					for f1 in val[0]:
						img = imread(f1)
						norm_image = img.astype(np.float32) * (1 - 0) / 255.0 + 0
						image_batch.append(norm_image)
					image_batch = np.array(image_batch)
					image_batch = np.expand_dims(image_batch,axis=-1)
					output_img = sess.run(pred,feed_dict={input_image:image_batch})
					op,loss = sess.run([train_loss_OP, train_loss],feed_dict={input_image:image_batch,output_mask:output_img})
					print("Training Loss",loss)
					train_loss_list.append(loss)


				combindedValDataset = tf.data.Dataset.zip((xval_dataset, yval_dataset)).shuffle(X_val.shape[0]).batch(batch_size)
				iterator = combindedValDataset.make_initializable_iterator()
				next_element = iterator.get_next()
				sess.run(iterator.initializer)
				for batches1 in range(2):
					image_batch1 = []
					val = sess.run(next_element)
					for f2 in val[0]:
						img = imread(f2)
						norm_image = img.astype(np.float32) * (1 - 0) / 255.0 + 0
						image_batch1.append(norm_image)
					image_batch1 = np.array(image_batch1)
					image_batch1 = np.expand_dims(image_batch1,axis=-1)
					output_img1 = sess.run(pred,feed_dict={input_image:image_batch1})
					val_loss = sess.run(train_loss,feed_dict={input_image:image_batch1,output_mask:output_img1})
					print("Validation Loss",val_loss)
					val_loss_list.append(val_loss)

				if((epochs>1) and (epochs%25 == 0)):
				#if(epochs == 0):	
					os.makedirs(output_dir+'/Output'+str(epochs))
					save_path1 = saver.save(sess, output_dir+"/Output"+str(epochs)+"/saved_model"+str(epochs)+".ckpt")
					print("Model saved after 25 epochs in path: %s" % save_path1)


		plt.figure()
		plt.plot(loss)
		plt.plot(val_loss)
		plt.savefig(output_dir+'/Loss.png')
		plt.close()
	









	



















