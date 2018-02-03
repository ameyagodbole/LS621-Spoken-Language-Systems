import numpy as np
import matplotlib.pyplot as plt

def dtw(train_spec, test_spec, show_img=False, cmn=False):
	'''
	DTW for P = 1/2 (refer Sakoe and Chiba (Dynamic Programming Algorithm Optimisation for Spoken Word Recognition))
	
	Arguments
	train_spec: training sample
	test_spec: sample to be compared with
	show_img: whether to plot frame similarity matrix and best path
	cmn: perform cepstral mean normalization
	Returns
	cost of best path, direction of best path
	'''
	if cmn:
		# cepstral mean normalization
		one = (train_spec - np.mean(train_spec, axis=0, keepdims=True))/np.var(train_spec, axis=0, keepdims=True)
		two = (test_spec - np.mean(test_spec, axis=0, keepdims=True))/np.var(test_spec, axis=0, keepdims=True)
	else:
		one = train_spec
		two = test_spec

	# using numpy broadcasting for faster execution
	one = np.expand_dims(one, axis=1)
	two = np.expand_dims(two, axis=0)
	
	sm_matrix = np.sqrt(np.sum((one-two)**2, axis=2))
	
	# store path cost
	dp_mat = np.zeros(sm_matrix.shape, dtype=np.float32)
	# store direction taken at each point: 0=down, 1=down+right, 2=right
	dp_dir = {}
	
	for i in reversed(range(dp_mat.shape[0])):
		for j in reversed(range(dp_mat.shape[1])):
			if (i==dp_mat.shape[0]-1) and (j==dp_mat.shape[1]-1):
				dp_mat[i,j] = sm_matrix[i,j]
				dp_dir[(i,j)] = [(i,j)]
			elif i==dp_mat.shape[0]-1:
				if j>=dp_mat.shape[1]-3:
					dp_mat[i,j] = sm_matrix[i,j] + dp_mat[i, j+1]
					dp_dir[(i,j)] = dp_dir[(i,j+1)] + [(i,j)]
				else:
					dp_mat[i,j] = np.inf
					dp_dir[(i,j)] = None
			elif j==dp_mat.shape[1]-1:
				if i>=dp_mat.shape[0]-3:
					dp_mat[i,j] = sm_matrix[i,j] + dp_mat[i+1, j]
					dp_dir[(i,j)] = dp_dir[(i+1,j)] + [(i,j)]
				else:
					dp_mat[i,j] = np.inf
					dp_dir[(i,j)] = None
			else:
				dp_mat[i,j] = np.inf
				dp_dir[(i,j)] = None
				candidate_val = []
				candidate_path = []
				if i+1<dp_mat.shape[0] and j+3<dp_mat.shape[1] and dp_dir[(i+1, j+3)] is not None:
					candidate_val.append(sm_matrix[i,j] + sm_matrix[i,j+1] + 2*sm_matrix[i,j+2] + dp_mat[i+1,j+3])
					candidate_path.append(dp_dir[(i+1,j+3)]+[(i,j+2),(i,j+1),(i,j)])
				if i+1<dp_mat.shape[0] and j+2<dp_mat.shape[1] and dp_dir[(i+1, j+2)] is not None:
					candidate_val.append(sm_matrix[i,j] + 2*sm_matrix[i,j+1] + dp_mat[i+1,j+2])
					candidate_path.append(dp_dir[(i+1,j+2)]+[(i,j+1),(i,j)])
				if i+2<dp_mat.shape[0] and j+1<dp_mat.shape[1] and dp_dir[(i+2, j+1)] is not None:
					candidate_val.append(sm_matrix[i,j] + 2*sm_matrix[i+1,j] + dp_mat[i+2,j+1])
					candidate_path.append(dp_dir[(i+2,j+1)]+[(i+1,j),(i,j)])            
				if i+3<dp_mat.shape[0] and j+1<dp_mat.shape[1] and dp_dir[(i+3, j+1)] is not None:
					candidate_val.append(sm_matrix[i,j] + sm_matrix[i+1,j] + 2*sm_matrix[i+2,j] + dp_mat[i+3,j+1])
					candidate_path.append(dp_dir[(i+3,j+1)]+[(i+2,j),(i+1,j),(i,j)])
				if dp_dir[(i+1, j+1)] is not None:
					candidate_val.append(2*sm_matrix[i,j] + dp_mat[i+1,j+1])
					candidate_path.append(dp_dir[(i+1,j+1)]+[(i,j)])
				if len(candidate_val)>0:
					dp_mat[i,j] = min(candidate_val)
					dp_dir[(i,j)] = candidate_path[np.argmin(candidate_val)]
	
	# get best path
	dp_dir[(0,0)].reverse()
	x,y = zip(*dp_dir[(0,0)])
	
	if show_img:
		#print 'cost: {:,}'.format(dp_mat[0,0])
		#print 'scaled difference matrix'
		plt.imshow(sm_matrix/np.max(sm_matrix))
		plt.colorbar()
		plt.axes().xaxis.set_tick_params(labeltop='on')        
		plt.plot(y,x)
		plt.xlabel('test_sample')
		plt.ylabel('train_sample')
		plt.show()
		
	return dp_mat[0,0]/float(dp_mat.shape[0]+dp_mat.shape[1]), zip(x,y)