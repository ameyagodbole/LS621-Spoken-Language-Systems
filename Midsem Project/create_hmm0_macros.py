#!/usr/bin/python

import sys
import os

if len(sys.argv)<3:
	raise ValueError("Usage: python create_hmm0_macros.py macros_file hmm0_vFloors_file")

macros_file = sys.argv[1]
hmm0_vFloors_file = sys.argv[2]

out = '~o\n\t<VECSIZE> 39\n\t<MFCC_0_D_A>\n'

with open(macros_file,'w') as f:
	f.write(out)

cmd = 'cat {} >> {}'.format(hmm0_vFloors_file,macros_file)
os.system(cmd)