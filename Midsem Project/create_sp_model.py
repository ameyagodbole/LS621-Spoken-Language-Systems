#!/usr/bin/python

import sys
import os

if len(sys.argv)<3:
	raise ValueError("Usage: python create_sp_model.py src_defs_file target_defs_file")

src_defs_file = sys.argv[1]
target_defs_file = sys.argv[2]

out = '~h "sp"\n<BEGINHMM>\n<NUMSTATES> 3\n<STATE> 2\n'
with open(src_defs_file, 'r') as f:
	in_sil = False
	in_s3 = False
	for line in f.readlines():
		if not in_sil and line.strip() != '~h "sil"':
			continue
		if not in_sil and line.strip() == '~h "sil"':
			in_sil = True
			print 'found "sil"'
			continue
		if in_sil and not in_s3 and line.strip()!='<STATE> 3':
			continue
		if in_sil and not in_s3 and line.strip()=='<STATE> 3':
			in_s3 = True
			print 'found state 3'
			continue
		if in_s3 and line.strip()!='<STATE> 4':
			out += line
			continue
		if in_s3 and line.strip()=='<STATE> 4':
			break

out += '<TRANSP> 3\n0.0 1.0 0.0\n 0.0 0.9 0.1\n 0.0 0.0 0.0\n<ENDHMM>\n'

with open(src_defs_file, 'r') as f:
	out = ''.join(f.readlines()) + out

with open(target_defs_file, 'w') as f:
	f.write(out)