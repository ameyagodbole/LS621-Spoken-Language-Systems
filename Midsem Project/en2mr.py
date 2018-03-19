#!/usr/bin/python

import sys

if len(sys.argv) != 4:
	raise ValueError("Usage: python en2mr.py input.txt en2mr_dict output.txt")

input_file = sys.argv[1]
en2mr_dict_file = sys.argv[2]
output_file = sys.argv[3]

en2mr_dict = {}
with open(en2mr_dict_file, 'r') as f:
	for line in f.readlines():
		ln = line.strip().split()
		en2mr_dict[ln[0]] = ln[1]

out = ''
idx = 0
with open(input_file, 'r') as f:
	for line in f.readlines():
		idx += 1
		ln = line.strip().split()
		out_line = '*/sample{} '.format(str(idx).zfill(2)) + ' '.join([en2mr_dict[w] for w in ln[1:]])
		out +=  out_line
		out += '\n'

with open(output_file, 'w') as f:
	f.write(out)