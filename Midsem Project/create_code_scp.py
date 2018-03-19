#!/usr/bin/python

import sys
import os
from glob import glob
from natsort import natsorted

if len(sys.argv)<5:
	raise ValueError("Usage: python create_code_scp.py.py scp_file source_folder_prefix source_file_prefix target_folder_prefix")

scp_file = sys.argv[1]
source_folder_prefix = sys.argv[2]
source_file_prefix = sys.argv[3]
target_folder_prefix = sys.argv[4]

out = ''

source_folders = natsorted(glob(source_folder_prefix+'*'))
for folder in source_folders:
	target_folder = folder.replace(source_folder_prefix, target_folder_prefix, 1)
	if not os.path.isdir(target_folder):
		os.makedirs(target_folder)
	
	source_files = natsorted(glob(os.path.join(folder, source_file_prefix+'*.wav')))
	for sfile in source_files:
		out += os.path.abspath(sfile)
		out += ' '
		tfile =  '.'.join(os.path.basename(sfile).split('.')[:-1])+'.mfc'
		out += os.path.abspath(os.path.join(target_folder,tfile))
		out += '\n'

with open(scp_file, 'w') as f:
	f.write(out)