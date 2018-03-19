#!/usr/bin/python

import sys
import os

if len(sys.argv) < 2 or len(sys.argv)>3:
	raise ValueError("Usage: python record_audio.py prompts.txt [langId]")

prompts_file = sys.argv[1]

if len(sys.argv)==3:
	langId = sys.argv[2]
else:
	langId = 'MR'

nsec = 4

spkrId = -1
while spkrId<0:
	spkrId = int(raw_input("Enter the speaker Id (Ex: 5): "))

fname_prefix = raw_input("Enter file name prefix [default is sample]: ")
if fname_prefix == '':
	fname_prefix = 'sample'

start_sentIdx = 0
while start_sentIdx<=0:
	start_sentIdx = int(raw_input("Enter the serial number (Ex: 11) of the first sentence to be recorded in this recording session: "))

with open(prompts_file, 'r') as f:
	lines = f.readlines()

if len(lines) < start_sentIdx:
	raise ValueError("Index of first sentence to record is greater than number of prompts ({})".format(len(lines)))

print "Type control c to terminate the recording session at any time"

out_dir = './wav{}_{}'.format(langId,spkrId)

if not os.path.isdir(out_dir):
	os.mkdir(out_dir)

for sentIdx in range(start_sentIdx-1, len(lines)):
	line = lines[sentIdx].strip()
	fname = os.path.join(out_dir, '{}{}.wav'.format(fname_prefix,str(sentIdx+1).zfill(2)))
	lab_fname = os.path.join(out_dir, '{}{}.lab'.format(fname_prefix,str(sentIdx+1).zfill(2)))
	
	print line
	raw_input("Press enter and read sentence (Number = {}) within {} seconds: ".format(sentIdx+1, nsec))
	os.system('rec -V1 -r 16000 -b 16 -c 1 {} trim 0 {}'.format(fname, nsec))

	with open(lab_fname, 'w') as f:
		f.write(' '.join(line.split()[1:]).upper())

print "Thanks; the recording session is over"