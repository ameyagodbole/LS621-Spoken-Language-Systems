#!/usr/bin/python

import sys

if len(sys.argv)<2:
	raise ValueError("Usage: python promt_file_pp.py promtps.txt")

prompts_file = sys.argv[1] 
with open(prompts_file, 'r') as f:
	lines = f.readlines()

out = ''
for line in lines:
	ln = line.strip().split()
	out += '*/test' + ln[0].split('.')[0].zfill(2) + ' '
	out += ' '.join(ln[1:])
	out += '\n'

with open(prompts_file, 'w') as f:
	f.write(out)