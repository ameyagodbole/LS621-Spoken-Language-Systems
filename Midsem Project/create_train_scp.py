#!/usr/bin/python

import sys
import os
from glob import glob
from natsort import natsorted

if len(sys.argv)<4:
	raise ValueError("Usage: python create_train_scp.py scp_file source_folder_prefix source_file_prefix")

scp_file = sys.argv[1]
source_folder_prefix = sys.argv[2]
source_file_prefix = sys.argv[3]

out = ''

source_folders = natsorted(glob(source_folder_prefix+'*'))
for folder in source_folders:
	source_files = natsorted(glob(os.path.join(folder, source_file_prefix+'*.mfc')))
	for sfile in source_files:
		out += os.path.abspath(sfile)
		out += '\n'

with open(scp_file, 'w') as f:
	f.write(out)