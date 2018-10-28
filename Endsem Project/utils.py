import numpy as np
import pandas as pd
import os
import h5py
import pickle
import scipy.io.wavfile
from glob import glob
from natsort import natsorted
from random import shuffle

def get_phone_dict(label_filename_fmt):
	"""
	Arguments
	label_filename_fmt: Filename convention to pass to glob
	---
	Return
	phone_dict: Dictionary mappng all phones to a number (can be used for generating one-hot label)
	"""
	all_lab_files = glob(label_filename_fmt)
	all_lab_files = natsorted(all_lab_files)
	phone_dict = {}
	phone_dict['x'] = 0
	phone_dict['pau'] = 1
	count = 2

	for lab_file in all_lab_files:
		with open(lab_file, 'r') as f:
			for line in f.readlines():
				label = line.strip().split()[-1]
				p1, rest = label.split('^',1)
				p2, rest = rest.split('-',1)
				p3, rest = rest.split('+',1)
				p4, rest = rest.split('=',1)
				p5_p6, rest = rest.split('_',1)
				p5, p6 = p5_p6.rsplit('@',1)

				if p1 not in phone_dict.keys():
					phone_dict[p1] = count
					count += 1
				if p2 not in phone_dict.keys():
					phone_dict[p2] = count
					count += 1
				if p3 not in phone_dict.keys():
					phone_dict[p3] = count
					count += 1
				if p4 not in phone_dict.keys():
					phone_dict[p4] = count
					count += 1
				if p5 not in phone_dict.keys():
					phone_dict[p5] = count
					count += 1

	return phone_dict

def parse_label_files(label_filename_fmt, output_directory):
	"""
	Arguments
	label_filename_fmt: Filename convention to pass to glob
	output_directory: The directory to save the output csv files to
	"""
	all_lab_files = glob(label_filename_fmt)
	all_lab_files = natsorted(all_lab_files)

	for lab_file in all_lab_files:
		df = pd.DataFrame(columns=['t1','t2','p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'c1', 'c2', 'c3', 'd1', 'd2', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'f1', 'f2', 'g1', 'g2', 'h1', 'h2', 'h3', 'h4', 'h5', 'i1', 'i2', 'j1', 'j2', 'j3'])
		idx = 0

		with open(lab_file, 'r') as f:
			for line in f.readlines():
				t1, t2, label = line.strip().split()
				p1, rest = label.split('^',1)
				p2, rest = rest.split('-',1)
				p3, rest = rest.split('+',1)
				p4, rest = rest.split('=',1)
				p5_p6, rest = rest.split('_',1)
				p5, p6 = p5_p6.rsplit('@',1)
				p7, rest = rest.split('/',1)
				
				_, rest = rest.split(':',1)
				a1, rest = rest.split('_',1)
				a2, rest = rest.split('_',1)
				a3, rest = rest.split('/',1)

				_, rest = rest.split(':',1)
				b1, rest = rest.split('-',1)
				b2, rest = rest.split('-',1)
				b3, rest = rest.split('@',1)
				b4, rest = rest.split('-',1)
				b5, rest = rest.split('&',1)
				b6, rest = rest.split('-',1)
				b7, rest = rest.split('#',1)
				b8, rest = rest.split('-',1)
				b9, rest = rest.split('$',1)
				b10, rest = rest.split('-',1)
				b11, rest = rest.split('!',1)
				b12, rest = rest.split('-',1)
				b13, rest = rest.split(';',1)
				b14, rest = rest.split('-',1)
				b15, rest = rest.split('|',1)
				b16, rest = rest.split('/',1)

				_, rest = rest.split(':',1)
				c1, rest = rest.split('+',1)
				c2, rest = rest.split('+',1)
				c3, rest = rest.split('/',1)

				_, rest = rest.split(':',1)
				d1, rest = rest.split('_',1)
				d2, rest = rest.split('/',1)

				_, rest = rest.split(':',1)
				e1, rest = rest.split('+',1)
				e2, rest = rest.split('@',1)
				e3, rest = rest.split('+',1)
				e4, rest = rest.split('&',1)
				e5, rest = rest.split('+',1)
				e6, rest = rest.split('#',1)
				e7, rest = rest.split('+',1)
				e8, rest = rest.split('/',1)

				_, rest = rest.split(':',1)
				f1, rest = rest.split('_',1)
				f2, rest = rest.split('/',1)

				_, rest = rest.split(':',1)
				g1, rest = rest.split('_',1)
				g2, rest = rest.split('/',1)

				_, rest = rest.split(':',1)
				h1, rest = rest.split('=',1)
				h2, rest = rest.split('@',1)	# slightly different from the convention
				h3, rest = rest.split('=',1)
				h4, rest = rest.split('|',1)
				h5, rest = rest.split('/',1)

				_, rest = rest.split(':',1)
				i1, rest = rest.split('=',1)
				i2, rest = rest.split('/',1)

				_, rest = rest.split(':',1)
				j1, rest = rest.split('+',1)
				j2, j3 = rest.split('-',1)

				df_row = pd.DataFrame({'t1':t1,'t2':t2,'p1':p1, 'p2':p2, 'p3':p3, 'p4':p4, 'p5':p5, 'p6':p6, 'p7':p7, 'a1':a1, 'a2':a2, 'a3':a3, 'b1':b1, 'b2':b2, 'b3':b3, 'b4':b4, 'b5':b5, 'b6':b6, 'b7':b7, 'b8':b8, 'b9':b9, 'b10':b10, 'b11':b11, 'b12':b12, 'b13':b13, 'b14':b14, 'b15':b15, 'b16':b16, 'c1':c1, 'c2':c2, 'c3':c3, 'd1':d1, 'd2':d2, 'e1':e1, 'e2':e2, 'e3':e3, 'e4':e4, 'e5':e5, 'e6':e6, 'e7':e7, 'e8':e8, 'f1':f1, 'f2':f2, 'g1':g1, 'g2':g2, 'h1':h1, 'h2':h2, 'h3':h3, 'h4':h4, 'h5':h5, 'i1':i1, 'i2':i2, 'j1':j1, 'j2':j2, 'j3':j3}, index=[idx])
				idx += 1
				df = df.append(df_row)

		save_file = os.path.join(output_directory, os.path.splitext(os.path.basename(lab_file))[0]+'.csv')
		# print 'writing label from:',lab_file,'to:',save_file
		df = df[['t1','t2','p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'c1', 'c2', 'c3', 'd1', 'd2', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'f1', 'f2', 'g1', 'g2', 'h1', 'h2', 'h3', 'h4', 'h5', 'i1', 'i2', 'j1', 'j2', 'j3']]
		df.to_csv(save_file, index=False)

def create_data_scp(filename_fmt,wav_folder='rwavs',lab_folder='csv_labs',f0_folder='f0',test_samples=30):
	"""
	Arguments
	filename_fmt: Format for WAV files
	wav_folder: Name of folder containing the WAV files
	lab_folder: Name of folder containing the csv label files
	f0_folder: Name of folder containing the F0 files
	test_samples: Number of samples from each folder to hold out as test set
	---
	Return
	Saves scripts to file train_list.scp and test_list.scp
	"""
	all_wav_files = glob(filename_fmt)
	if test_samples>0:
		shuffle(all_wav_files)
		train_wav_files = all_wav_files[:-test_samples]
		test_wav_files = all_wav_files[-test_samples:]
	else:
		train_wav_files = all_wav_files
	train = open('train_list.scp','w')
	for wav_file in train_wav_files:
		par_dir, part_no = os.path.split(os.path.dirname(wav_file))
		par_dir = os.path.dirname(par_dir)
		lab_file = os.path.join(par_dir, lab_folder, part_no, os.path.splitext(os.path.basename(wav_file))[0]+'.csv')
		f0_file = os.path.join(par_dir, f0_folder, part_no, os.path.splitext(os.path.basename(wav_file))[0]+'.txt')
		train.write('{} {} {}\n'.format(wav_file, lab_file, f0_file))
	train.close()

	if test_samples>0:
		test = open('test_list.scp','w')
		for wav_file in test_wav_files:
			par_dir, part_no = os.path.split(os.path.dirname(wav_file))
			par_dir = os.path.dirname(par_dir)
			lab_file = os.path.join(par_dir, lab_folder, part_no, os.path.splitext(os.path.basename(wav_file))[0]+'.csv')
			f0_file = os.path.join(par_dir, f0_folder, part_no, os.path.splitext(os.path.basename(wav_file))[0]+'.txt')
			test.write('{} {} {}\n'.format(wav_file, lab_file, f0_file))
		test.close()

def preprocess_input(file_scp, phone_map_pkl, pos_map_pkl, output_file, inp_dim=None, padding = 1024, pad_row=None, freq = 16000):
	"""
	Arguments
	file_scp: Line containing (space separated) wav_filenm lab_filenm f0_filenm 
	phone_map_pkl: File containing the phone to id map
	pos_map_pkl: File containing the POS to id map
	output_file: The output csv file
	inp_dim: The dimension of the input
	padding: The padding data to add between files 
	freq: The frequency at which the samples should be recorded
	"""
	if not inp_dim:
		inp_dim = 406

	wav_scale = max(np.abs(np.iinfo(np.int16).min),np.iinfo(np.int16).max)
	
	with open(phone_map_pkl,'r') as f:
		phone_map = pickle.load(f)

	with open(pos_map_pkl,'r') as f:
		pos_map = pickle.load(f)

	def expand_row(row, phone_map, pos_map):
		x = np.zeros(inp_dim)
		offset = 0
		
		x[phone_map[str(row['p1'])]]=1
		offset += len(phone_map)

		x[offset+phone_map[str(row['p2'])]]=1
		offset += len(phone_map)

		x[offset+phone_map[str(row['p3'])]]=1
		offset += len(phone_map)

		x[offset+phone_map[str(row['p4'])]]=1
		offset += len(phone_map)
		
		x[offset+phone_map[str(row['p5'])]]=1
		offset += len(phone_map)
		
		x[offset]=0 if row['p6']=='x' else int(row['p6'])
		offset += 1
		
		x[offset]=0 if row['p6']=='x' else int(row['p7'])
		offset += 1
		
		x[offset] = 1 if row['a1']==1 else 0
		offset += 1

		x[offset] = 1 if row['a2']==1 else 0
		offset += 1

		x[offset] = row['a3']
		offset += 1

		x[offset] = 0 if row['b1']=='x' else int(row['b1'])
		offset += 1

		x[offset] = 0 if row['b2']=='x' else int(row['b2'])
		offset += 1

		x[offset] = 0 if row['b3']=='x' else int(row['b3'])
		offset += 1

		x[offset] = 0 if row['b4']=='x' else int(row['b4'])
		offset += 1

		x[offset] = 0 if row['b5']=='x' else int(row['b5'])
		offset += 1

		x[offset] = 0 if row['b6']=='x' else int(row['b6'])
		offset += 1

		x[offset] = 0 if row['b7']=='x' else int(row['b7'])
		offset += 1

		x[offset] = 0 if row['b8']=='x' else int(row['b8'])
		offset += 1

		x[offset] = 0 if row['b9']=='x' else int(row['b9'])
		offset += 1

		x[offset] = 0 if row['b10']=='x' else int(row['b10'])
		offset += 1

		x[offset] = 0 if row['b11']=='x' else int(row['b11'])
		offset += 1

		x[offset] = 0 if row['b12']=='x' else int(row['b12'])
		offset += 1

		x[offset] = 0 if row['b13']=='x' else int(row['b13'])
		offset += 1

		x[offset] = 0 if row['b14']=='x' else int(row['b14'])
		offset += 1

		x[offset] = 0 if row['b15']=='x' else int(row['b15'])
		offset += 1

		x[offset + phone_map[str(row['b16'])]] = 1
		offset += len(phone_map)

		x[offset] = 0 if row['c1']=='x' else int(row['c1'])
		offset += 1

		x[offset] = 0 if row['c2']=='x' else int(row['c2'])
		offset += 1

		x[offset] = 0 if row['c3']=='x' else int(row['c3'])
		offset += 1

		x[offset + pos_map[str(row['d1'])]] = 1
		offset += len(pos_map)

		x[offset] = 0 if row['d2']=='x' else int(row['d2'])
		offset += 1

		x[offset + pos_map[str(row['e1'])]] = 1
		offset += len(pos_map)

		x[offset] = 0 if row['e2']=='x' else int(row['e2'])
		offset += 1

		x[offset] = 0 if row['e3']=='x' else int(row['e3'])
		offset += 1

		x[offset] = 0 if row['e4']=='x' else int(row['e4'])
		offset += 1

		x[offset] = 0 if row['e5']=='x' else int(row['e5'])
		offset += 1

		x[offset] = 0 if row['e6']=='x' else int(row['e6'])
		offset += 1

		x[offset] = 0 if row['e7']=='x' else int(row['e7'])
		offset += 1

		x[offset] = 0 if row['e8']=='x' else int(row['e8'])
		offset += 1

		x[offset + pos_map[str(row['f1'])]] = 1
		offset += len(pos_map)

		x[offset] = 0 if row['f2']=='x' else int(row['f2'])
		offset += 1

		x[offset] = 0 if row['g1']=='x' else int(row['g1'])
		offset += 1

		x[offset] = 0 if row['g2']=='x' else int(row['g2'])
		offset += 1

		x[offset] = 0 if row['h1']=='x' else int(row['h1'])
		offset += 1

		x[offset] = 0 if row['h2']=='x' else int(row['h2'])
		offset += 1

		x[offset] = 0 if row['h3']=='x' else int(row['h3'])
		offset += 1

		x[offset] = 0 if row['h4']=='x' else int(row['h4'])
		offset += 1

		x[offset] = 0 if row['i1']=='x' else int(row['i1'])
		offset += 1

		x[offset] = 0 if row['i2']=='x' else int(row['i2'])
		offset += 1

		x[offset] = 0 if row['j1']=='x' else int(row['j1'])
		offset += 1

		x[offset] = 0 if row['j2']=='x' else int(row['j2'])
		offset += 1

		x[offset] = 0 if row['j3']=='x' else int(row['j3'])
		offset += 1

		return x

	if not pad_row:
		pad_row_s = pd.Series(data={'p1':'x', 'p2':'x', 'p3':'x', 'p4':'x', 'p5':'x', 'p6':'x', 'p7':'x', 'a1':0, 'a2':0, 'a3':0, 'b1':'x', 'b2':'x', 'b3':'x', 'b4':'x', 'b5':'x', 'b6':'x', 'b7':'x', 'b8':'x', 'b9':'x', 'b10':'x', 'b11':'x', 'b12':'x', 'b13':'x', 'b14':'x', 'b15':'x', 'b16':'x', 'c1':'x', 'c2':'x', 'c3':'x', 'd1':0, 'd2':0, 'e1':0, 'e2':'x', 'e3':'x', 'e4':'x', 'e5':'x', 'e6':'x', 'e7':'x', 'e8':'x', 'f1':0, 'f2':0, 'g1':0, 'g2':0, 'h1':'x', 'h2':'x', 'h3':1, 'h4':1, 'h5':0, 'i1':0, 'i2':0, 'j1':5, 'j2':5, 'j3':5})
		pad_row = expand_row(pad_row_s, phone_map, pos_map)

	with h5py.File(output_file, 'w') as D:
		dx = D.create_dataset('X', (1, 256+inp_dim+2), dtype=np.dtype('float16'), compression="gzip", maxshape=(None, 256+inp_dim+2))
		dy = D.create_dataset('Y', (1, 256), dtype=np.dtype('float16'), compression="gzip", maxshape=(None, 256))
		
		wav_filenm, lab_filenm, f0_filenm = file_scp.strip().split()
		rate, wav_data = scipy.io.wavfile.read(wav_filenm)
		rate = float(rate)
		# mu-law transformation
		wav_data = wav_data.astype(np.float32)
		wav_data /= wav_scale
		wav_data = np.sign(wav_data)*np.log(1+255*np.abs(wav_data))/np.log(256)
		wav_data = (wav_data+1)*128
		wav_data = np.floor(wav_data).astype(np.int16)
		W = np.zeros((len(wav_data),256), dtype=np.uint8)
		W[range(len(wav_data)), wav_data] = 1
		
		# sample label file
		X = np.zeros((len(wav_data),inp_dim),dtype=np.uint8)
		lab_df = pd.read_csv(lab_filenm)
		r = 0
		r_changed = False
		for x in range(len(wav_data)):
			t_x = x*1./rate
			while r<len(lab_df)-1 and t_x > lab_df.loc[r,'t2']:
				r_changed = True
				r += 1
			if r_changed or x==0:
				X[x,:] = expand_row(lab_df.loc[r], phone_map, pos_map)
			else:
				X[x,:] = X[x-1,:]
			r_changed = False

		# sample F0 file
		F = np.zeros((len(wav_data),2),dtype=np.float16)
		f0_df = pd.read_csv(f0_filenm, sep='\t')
		r = 0
		for x in range(len(wav_data)):
			t_x = x*1./rate
			while r<len(f0_df)-1 and t_x > f0_df.loc[r,'time']:
				r += 1
			F[x,0] = 0 if f0_df.loc[r,'pitch']=='--undefined--' else 1
			F[x,1] = np.log(float(f0_df.loc[r,'pitch'])) if F[x,0] else 0
		
		dx.resize((len(wav_data) + padding, 256+inp_dim+2))
		dy.resize((len(wav_data) + padding, 256))
		dx[:len(wav_data) + padding, :] = 0
		dx[:len(wav_data) + padding, 128] = 1
		dx[:padding, 256:256+inp_dim] = pad_row
		dx[padding:padding+len(wav_data), 0:256] = W
		dx[padding-1:padding+len(wav_data)-1, 256:256+inp_dim] = X
		dx[padding-1:padding+len(wav_data)-1, 256+inp_dim:256+inp_dim+2] = F

		dy[:len(wav_data) + padding, :] = 0
		dy[:len(wav_data) + padding, 128] = 1
		dy[padding-1:padding+len(wav_data)-1, 0:256] = W

def hdf5matrix(file, dset):
	"""
	Arguments
	file: The hdf5 data file
	dset: Name of the dataset inside the file
	---
	Return
	Dataset as a Numpy matrix
	"""
	f = h5py.File(file, 'r')
	data = f[dset].value
	f.close()
	return data