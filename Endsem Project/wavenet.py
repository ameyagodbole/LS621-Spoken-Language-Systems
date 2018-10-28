import numpy as np
import os.path as path
from keras.layers import Input, Conv1D, Add, Multiply, Activation, Lambda, Dense
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from scipy.io.wavfile import write as wav_write
from timeseriesgenerator import TimeseriesGenerator
from utils import hdf5matrix

class WaveNet(object):
	def __init__(self, input_dim, dilations, filter_width=2):
		self.input_dim = input_dim
		self.dilations = dilations
		self.filter_width = filter_width
		self.receptive_field = (filter_width - 1) * np.sum(np.asarray(dilations)) + 1
		self.model = None
		self.callbacks = None
		super(WaveNet, self).__init__()

	def build(self):
		l = len(self.dilations)
		r = 64
		s = 256
		a = 256

		inputs = Input(shape=(self.receptive_field, self.input_dim,), name='input')
		h = {}
		skip = []

		h[0] = Conv1D(r, self.filter_width, strides=1, padding='causal', dilation_rate=self.dilations[0],
		 activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='h0')(inputs)

		for idx,d in enumerate(self.dilations):
			if idx == 0:
				continue
			# gated activation
			tan_out = Conv1D(r, self.filter_width, strides=1, padding='causal', dilation_rate=d, activation='tanh', 
				use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='tan_{}'.format(idx))(h[idx-1])
			sigmoid_out = Conv1D(r, self.filter_width, strides=1, padding='causal', dilation_rate=d, activation='sigmoid', 
				use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='sig_{}'.format(idx))(h[idx-1])
			ts = Multiply(name='dot_{}'.format(idx))([tan_out, sigmoid_out])
			if idx<l-1:
				# residual connection
				ts_to_res = Conv1D(r, 1, strides=1, padding='same', dilation_rate=1, activation=None, use_bias=True, 
					kernel_initializer='glorot_uniform', bias_initializer='zeros', name='ts_to_res_{}'.format(idx))(ts)
				h[idx] = Add(name='h{}'.format(idx))([ts_to_res, h[idx-1]])
			# skip connection
			def crop_shape(input_shape):
				return (input_shape[0],input_shape[2])
			crop_layer = Lambda(lambda x: x[:,-1,:], output_shape=crop_shape, name='crop_{}'.format(idx))(ts)
			skip_conn = Dense(s, activation=None, use_bias=True, 
				kernel_initializer='glorot_uniform', bias_initializer='zeros', name='skip{}'.format(idx))(crop_layer)
			skip.append(skip_conn)

		joint_out = Add(name='join')(skip)
		joint_out = Activation('relu', name='join_relu')(joint_out)

		r1 = Dense(a, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
			bias_initializer='zeros', name='dense1')(joint_out)
		output = Dense(a, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', 
			bias_initializer='zeros', name='output')(r1)

		self.model = Model(inputs=inputs, outputs=output)

	def compile(self):
		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	def plot(self, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB'):
		plot_model(self.model, to_file=to_file, show_shapes=show_shapes, show_layer_names=show_layer_names, rankdir=rankdir)

	def add_callbacks(self, filepath=None, lr_decay_rate=None):
		if filepath:
			model_ckpt = ModelCheckpoint(filepath, monitor='accuracy', verbose=1)
			if self.callbacks:
				self.callbacks.append(model_ckpt)
			else:
				self.callbacks = [model_ckpt]
		if lr_decay_rate:
			lr_decay = LearningRateScheduler(lambda e,lr: lr*lr_decay_rate)
			if self.callbacks:
				self.callbacks.append(model_ckpt)
			else:
				self.callbacks = [lr_decay]

	def remove_callbacks(self):
		self.callbacks = None

	def fit_on_file(self, train_file, dset_X='X', dset_Y='Y', batch_size=20, initial_epoch=0):
		"""
		Arguments
		train_file: HDF5 file containing the training dataset
		dset_X: Name of training set features inside the HDF5 file
		dset_Y: Name of training set labels inside the HDF5 file
		"""
		X = hdf5matrix(train_file, dset_X)
		Y = hdf5matrix(train_file, dset_Y)
		data_gen = TimeseriesGenerator(X, Y, length=self.receptive_field, batch_size=batch_size)
		steps_per_epoch = len(data_gen)
		self.model.fit_generator(data_gen, steps_per_epoch=steps_per_epoch, callbacks=self.callbacks, initial_epoch=initial_epoch)

	def evaluate_on_file(self, eval_file, save_dir, dset_X='X', dset_Y='Y'):
		"""
		Arguments
		eval_file: HDF5 file containing the evaluation dataset
		save_dir: Directory to save the generated audio
		dset_X: Name of training set features inside the HDF5 file
		dset_Y: Name of training set labels inside the HDF5 file
		"""
		save_file = os.path.join(save_dir, os.path.splitext(os.path.basename(eval_file))[0]+'.wav')
		X = hdf5matrix(train_file, dset_X)
		Y = hdf5matrix(train_file, dset_Y)
		audio = []
		data_gen = TimeseriesGenerator(X, Y, length=self.receptive_field, batch_size=1)
		steps = len(data_gen)
		for step in range(steps):
			if step==0:
				batch_x, batch_y = data_gen[step]
				pred = self.model.predict(batch_x, batch_size=1, verbose=1)
				val = np.random.choice(range(256), p = pred[:])
				audio.append(val)
			else:
				batch_x, batch_y = data_gen[step]
				batch_x[0,-1,:256] = 0
				batch_x[0,-1,int(audio[-1])] = 1
				pred = self.model.predict(batch_x, batch_size=1, verbose=1)
				val = np.random.choice(range(256), p = pred[:])
				audio.append(val)
		def inverse_mu_law(quantized):
			# scale to [-1,1]
			quantized = np.asarray(quantized).astype(np.float32)
			quantized /= 128
			quantized -= 1
			wav_scale = max(np.abs(np.iinfo(np.int16).min),np.iinfo(np.int16).max)
			quantized *= wav_scale
			audio = quantized.astype(np.int16)
			return audio

		gen_audio = inverse_mu_law(audio)
		wav_write(save_file, 16000, gen_audio)

	def save(self, save_file):
		"""
		Arguments
		save_file: Model save file
		"""
		self.model.save(save_file)