import numpy as np

class TimeseriesGenerator:
	"""
	Generates data for Keras
	"""
	def __init__(self, X, Y, length, batch_size):
		self.X = X
		self.Y = Y
		self.length = length
		self.batch_size = batch_size
		self.xdim = self.X.shape[1:]
		self.ydim = self.Y.shape[1:]
		self.curr_id = None
		self.on_epoch_end()

	def __len__(self):
		"""
		Denotes the number of batches per epoch
		"""		
		return int(np.ceil((len(self.X)-self.length) / self.batch_size))

	def next(self):
		"""
		Generate one batch of data
		"""
		yrows = np.arange(self.curr_id, min(self.curr_id+self.batch_size, len(self.X)))

		batch_X = np.zeros((len(yrows),self.length)+tuple(self.xdim))
		batch_Y = self.Y[self.curr_id:min(self.curr_id+self.batch_size, len(self.X)), :]

		for ir,r in enumerate(yrows):
			batch_X[ir, :, :] = self.X[r-self.length:r, :]

		self.curr_id += self.batch_size
		if self.curr_id >= len(self.X):
			self.on_epoch_end()
		return batch_X, batch_Y

	def on_epoch_end(self):
		"""
		Updates indexes after each epoch
		"""
		self.curr_id = self.length