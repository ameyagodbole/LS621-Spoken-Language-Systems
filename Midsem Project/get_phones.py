#!/usr/bin/python

import sys

if len(sys.argv)<3:
	raise ValueError("Usage: python get_phones.py pronunciation_dict target_file_prefix")

pronunciation_dict_file = sys.argv[1]
target_file_prefix = sys.argv[2]

with open(pronunciation_dict_file, 'r') as f:
	pronunciation_dict = f.readlines()

phones = set()
for line in pronunciation_dict:
	ln = line.strip().split()
	for p in ln[1:]:
		if p[0] == '[':
			continue
		phones.add(p)

phones.add('sil')

# Without sp
out1 = open(target_file_prefix+'0','w')
# With sp
out2 = open(target_file_prefix+'1','w')

for p in phones:
	out2.write(p+'\n')
	if p != 'sp':
		out1.write(p+'\n')

out1.close()
out2.close()