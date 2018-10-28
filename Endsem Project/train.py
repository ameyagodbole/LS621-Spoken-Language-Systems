import multiprocessing
import logging
import time
import os
from time import time as T
from utils import preprocess_input
from wavenet import WaveNet

class Consumer(multiprocessing.Process):
	
	def __init__(self, task_queue, result_queue_tr, result_queue_te):
		multiprocessing.Process.__init__(self)
		self.task_queue = task_queue
		self.result_queue_tr = result_queue_tr
		self.result_queue_te = result_queue_te

	def run(self):
		proc_name = self.name
		while True:
			next_task = self.task_queue.get()
			if next_task is None:
				# Poison pill means shutdown
				print '%s: Exiting' % proc_name
				self.task_queue.task_done()
				break
			start = T()
			answer = next_task()
			end = T()
			print '%s : %s : %s' % (proc_name, next_task, end-start)
			self.task_queue.task_done()
			if next_task.task_type == 'train':
				self.result_queue_tr.put(answer)
			elif next_task.task_type == 'test':
				self.result_queue_te.put(answer)
			else:
				raise ValueError('Incorrect task type: {}'.format(next_task.task_type))
		return


class Task(object):
	def __init__(self, line, data_save_dir, task_type, task_id):
		self.line = line.strip()
		self.data_save_dir = data_save_dir
		self.task_type = task_type
		self.task_id = task_id
		h5_nm = os.path.splitext(os.path.basename(self.line.split()[0]))[0] + '.h5'
		self.output_file = os.path.join(self.data_save_dir, task_type, h5_nm)
	def __call__(self):
		preprocess_input(self.line, PHONE_MAP_PKL, POS_MAP_PKL, self.output_file)
		return self.output_file
	def __str__(self):
		return '%s : %d : %s' % (self.task_type, self.task_id, self.output_file)

if __name__ == '__main__':
	TRAIN_SCRIPT_FILE = './train_list.scp'
	TEST_SCRIPT_FILE = './test_list.scp'
	PHONE_MAP_PKL = './phone_map.pkl'
	POS_MAP_PKL = './pos_map.pkl' 
	DATA_SAVE_DIR = './data/'
	TEST_SAVE_DIR = './test_wav/'
	CKPT_PATH = './checkpoints'
	EPOCHS = 100
	
	# Establish communication queues
	tasks = multiprocessing.JoinableQueue()
	results_tr = multiprocessing.Queue()
	results_te = multiprocessing.Queue()
	logger = multiprocessing.log_to_stderr()
	logger.setLevel(logging.INFO)

	# Start consumers
	num_consumers = max(1, multiprocessing.cpu_count() - 2)
	print 'Creating %d consumers' % num_consumers
	consumers = [ Consumer(tasks, results_tr, results_te) for i in xrange(num_consumers) ]
	for w in consumers:
		w.start()
	
	with open(TRAIN_SCRIPT_FILE, 'r') as f:
		train_list = f.readlines()
		num_train = len(train_list)
	with open(TEST_SCRIPT_FILE, 'r') as f:
		test_list = f.readlines()
		num_test = len(f.readlines())

	# Enqueue jobs
	for i in range(num_train):
		tasks.put(Task(train_list[i], DATA_SAVE_DIR, 'train', i))
	
	for i in range(num_test):
		tasks.put(Task(test_list[i], DATA_SAVE_DIR, 'test', i))

	# Add a poison pill for each consumer
	for i in range(num_consumers):
		tasks.put(None)
	
	wvn = WaveNet(input_dim=256+406+2, dilations=[1,2,4,8,16,32,64,128,256,512], filter_width=2)
	wvn.build()
	wvn.compile()
	wvn.plot()
	wvn.add_callbacks(os.path.join(CKPT_PATH,'weights.epoch001.{epoch:02d}.hdf5'), None)

	# Start 1st epoch training
	num_jobs = num_train
	train_files = []
	train_times = []
	while num_jobs:
		f = results_tr.get()
		train_files.append(f)
		# Model training
		start = T()
		wvn.fit_on_file(f)
		end = T()
		train_times.append(end-start)
		num_jobs -= 1
	print 'Average training time (per file):',np.mean(train_times)

	# Start 1st epoch testing
	num_jobs = num_test
	test_files = []
	test_times = []
	if not os.path.exists(os.path.join(TEST_SAVE_DIR, 'audio0')):
		os.makedirs(os.path.join(TEST_SAVE_DIR, 'audio0'))
	while num_jobs:
		f = results_te.get()
		test_files.append(f)
		# Model testing
		start = T()
		wvn.evaluate_on_file(f, os.path.join(TEST_SAVE_DIR, 'audio0'))
		end = T()
		test_times.append(end-start)
		num_jobs -= 1
	print 'Average testing time (per file):',np.mean(test_times)

	wvn.remove_callbacks()
	wvn.add_callbacks(lr_decay_rate=0.9886)
	for e in range(1,EPOCHS):
		for f in train_files:
			# Train model
			wvn.fit_on_file(f, initial_epoch=e)
		wvn.save(os.path.join(CKPT_PATH,'weights.epoch{}.hdf5'.format(str(e).zfill(3))))
		if not os.path.exists(os.path.join(TEST_SAVE_DIR, 'audio{}'.format(e))):
			os.makedirs(os.path.join(TEST_SAVE_DIR, 'audio{}'.format(e)))
		for f in test_files:
			# Test model
			wvn.evaluate_on_file(f, os.path.join(TEST_SAVE_DIR, 'audio{}'.format(e)))

	# Wait for all of the tasks to finish
	tasks.join()
	