#!/usr/bin/python

import sys
import os

if len(sys.argv)<4:
	raise ValueError("Usage: python create_hmm0_defs.py defs_file proto_file monophones_file")

defs_file = sys.argv[1]
proto_file = sys.argv[2]
monophones_file = sys.argv[3]

with open(proto_file, 'r') as f:
	proto = f.readlines()
with open(monophones_file, 'r') as f:
	phones = f.readlines()


for idx,ln in enumerate(proto):
	if ln[:2] == '~h':
		proto = proto[idx+1:]
		break

out = ''
for p in phones:
	out += '~h "{}"\n'.format(p.strip())
	out += ''.join(proto)

with open(defs_file, 'w') as f:
	f.write(out)