import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
#from data_xueguan import *
from keras.utils.np_utils import to_categorical
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob
import scipy.io
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation


class myUnet(object):

	# def __init__(self, img_rows = 384, img_cols = 640):

	# 	self.img_rows = img_rows
	# 	self.img_cols = img_cols

	# def load_data(self):

	# 	mydata = dataProcess(self.img_rows, self.img_cols)
	# 	imgs_train, imgs_mask_train = mydata.load_train_data()
	# 	imgs_test = mydata.load_test_data()
	# 	return imgs_train, imgs_mask_train, imgs_test

	def __init__(self, img_rows = 256, img_cols = 256):

		self.img_rows = img_rows
		self.img_cols = img_cols
		#self.data_path = "./data/train_xueguan_0525_256_danboduan/image"
		#self.label_path = "./data/train_xueguan_0525_256_danboduan/label"
		#self.test_path = "./data/test_xueguan"
		#self.img_type = "png"
		self.data_path = "./train_xueguan_0516_256_940boduan/src"
		self.label_path = "./train_xueguan_0516_256_940boduan/label"
		self.test_path = "./kz2/test"
		self.img_type = "png"

	def load_data(self):
        
		#hw=(256,256)

		i = 0
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		imgs = glob.glob(self.data_path+"/*."+self.img_type)
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.img_rows,self.img_cols,1), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.img_rows,self.img_cols,1), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]

			img = load_img(self.data_path + "/" + midname,grayscale = True)
                        #data_img = scipy.io.loadmat(self.data_path + "/" + midname)
                        #img = data_img['src']
			#img = img.resize(hw)

			label = load_img(self.label_path + "/" + midname,grayscale = True)
                        #data_label = scipy.io.loadmat(self.label_path + "/" + midname)
                        #label = data_label['label']
			#label = label.resize(hw)

			img = img_to_array(img)
			label = img_to_array(label)
			imgdatas[i] = img
			imglabels[i] = label
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('loading training images done')
		imgs_train = imgdatas.astype('float32')
		imgs_mask_train = imglabels.astype('float32')
		imgs_train /= 255
		imgs_mask_train /= 255
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0

		i_test = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		imgs_test = glob.glob(self.test_path+"/*."+"bmp")
		print(len(imgs_test))
		imgdatas_test = np.ndarray((len(imgs_test),self.img_rows,self.img_cols,1), dtype=np.uint8)
		for imgname_test in imgs_test:
			midname_test = imgname_test[imgname_test.rindex("/")+1:]

			img_test = load_img(self.test_path + "/" + midname_test,grayscale = True)
                        #img_test = scipy.io.loadmat(self.test_path + "/" + midname_test)
                        #img_test = img_test['src'] 
			#img_test=img_test.resize(hw)

			img_test = img_to_array(img_test)
			imgdatas_test[i_test] = img_test
			i_test += 1
		print('loading test images done')
		imgs_test = imgdatas_test.astype('float32')
		imgs_test /= 255

		return imgs_train, imgs_mask_train, imgs_test

	def get_unet(self):

		inputs = Input((self.img_rows, self.img_cols, 1))

		
		'''
		unet with crop(because padding = valid) 

		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(inputs)
		print "conv1 shape:",conv1.shape
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
		print "conv1 shape:",conv1.shape
		crop1 = Cropping2D(cropping=((90,90),(90,90)))(conv1)
		print "crop1 shape:",crop1.shape
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print "pool1 shape:",pool1.shape

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
		print "conv2 shape:",conv2.shape
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
		print "conv2 shape:",conv2.shape
		crop2 = Cropping2D(cropping=((41,41),(41,41)))(conv2)
		print "crop2 shape:",crop2.shape
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print "pool2 shape:",pool2.shape

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
		print "conv3 shape:",conv3.shape
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
		print "conv3 shape:",conv3.shape
		crop3 = Cropping2D(cropping=((16,17),(16,17)))(conv3)
		print "crop3 shape:",crop3.shape
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print "pool3 shape:",pool3.shape

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		crop4 = Cropping2D(cropping=((4,4),(4,4)))(drop4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = merge([crop4,up6], mode = 'concat', concat_axis = 3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = merge([crop3,up7], mode = 'concat', concat_axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = merge([crop2,up8], mode = 'concat', concat_axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = merge([crop1,up9], mode = 'concat', concat_axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
		'''

		conv1_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
                BatchNorm1_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv1_1)
                ReLU1_1 = Activation('relu')(BatchNorm1_1)
		conv1_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU1_1)
		drop1_2 = Dropout(0)(conv1_2)
                Merge1 = merge([conv1_1,drop1_2], mode = 'concat', concat_axis = 3)
		pool1 = MaxPooling2D(pool_size=(2, 2))(Merge1)

		conv2_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
                BatchNorm2_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv2_1)
                ReLU2_1 = Activation('relu')(BatchNorm2_1)
		conv2_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU2_1)
		drop2_2 = Dropout(0)(conv2_2)
                Merge2 = merge([conv2_1,drop2_2], mode = 'concat', concat_axis = 3)
		pool2 = MaxPooling2D(pool_size=(2, 2))(Merge2)

		conv3_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
                BatchNorm3_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv3_1)
                ReLU3_1 = Activation('relu')(BatchNorm3_1)
		conv3_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU3_1)
		drop3_2 = Dropout(0)(conv3_2)
                Merge3 = merge([conv3_1,drop3_2], mode = 'concat', concat_axis = 3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(Merge3)

		conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
                BatchNorm4_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv4_1)
                ReLU4_1 = Activation('relu')(BatchNorm4_1)
		conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU4_1)
		drop4_2 = Dropout(0)(conv4_2)
                Merge4 = merge([conv4_1,drop4_2], mode = 'concat', concat_axis = 3)
		drop4 = Dropout(0.5)(Merge4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5_1 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
                BatchNorm5_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv5_1)
                ReLU5_1 = Activation('relu')(BatchNorm5_1)
		conv5_2 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU5_1)
		drop5_2 = Dropout(0)(conv5_2)
                Merge5 = merge([conv5_1,drop5_2], mode = 'concat', concat_axis = 3)
		drop5 = Dropout(0.5)(Merge5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
		conv6_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
                BatchNorm6_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv6_1)
                ReLU6_1 = Activation('relu')(BatchNorm6_1)
		conv6_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU6_1)
		drop6_2 = Dropout(0)(conv6_2)
                Merge6 = merge([conv6_1,drop6_2], mode = 'concat', concat_axis = 3)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge6))
		merge7 = merge([Merge3,up7], mode = 'concat', concat_axis = 3)
		conv7_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
                BatchNorm7_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv7_1)
                ReLU7_1 = Activation('relu')(BatchNorm7_1)
		conv7_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU7_1)
		drop7_2 = Dropout(0)(conv7_2)
                Merge7 = merge([conv7_1,drop7_2], mode = 'concat', concat_axis = 3)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge7))
		merge8 = merge([Merge2,up8], mode = 'concat', concat_axis = 3)
		conv8_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
                BatchNorm8_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv8_1)
                ReLU8_1 = Activation('relu')(BatchNorm8_1)
		conv8_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU8_1)
		drop8_2 = Dropout(0)(conv8_2)
                Merge8 = merge([conv8_1,drop8_2], mode = 'concat', concat_axis = 3)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge8))
		merge9 = merge([Merge1,up9], mode = 'concat', concat_axis = 3)
		conv9_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
                BatchNorm9_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv9_1)
                ReLU9_1 = Activation('relu')(BatchNorm9_1)
		conv9_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU9_1)
		drop9_2 = Dropout(0)(conv9_2)
                Merge9 = merge([conv9_1,drop9_2], mode = 'concat', concat_axis = 3)

		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)#sigmoid
                #conv10 = Conv2D(1, 1, activation = 'softmax')(conv9)#sigmoid

		model = Model(input = inputs, output = conv10)

		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
                #model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

		return model


	def train(self):

		print("loading data")
		imgs_train, imgs_mask_train, imgs_test= self.load_data()

		print("loading data done")
		model = self.get_unet()
		print("got unet")

		model_checkpoint = ModelCheckpoint('unet_xueguan_0516_940_1223_256_300.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=300, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
                model.save()
                #model.fit(imgs_train, imgs_mask_train_binary, batch_size=4, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

		#print('predict test data')
		#imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		##np.save('./results_xueguan_0525_256_danboduan_densenet/imgs_mask_test.npy', imgs_mask_test)
		#np.save('./results/imgs_mask_test_densenet.npy', imgs_mask_test)

	def save_img(self):

		print("array to image")
		#imgs = np.load('./tttt/imgs_mask_test.npy')
		imgs = np.load('./results/imgs_mask_test_densenet.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save("./results/densenet/%d.jpg"%(i))




if __name__ == '__main__':
	myunet = myUnet()
	myunet.train()
	#myunet.save_img()








