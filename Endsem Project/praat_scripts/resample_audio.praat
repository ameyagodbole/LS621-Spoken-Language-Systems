# EDITED
# source
# https://github.com/FieldDB/Praat-Scripts/blob/master/change_sample_rate_of_sound_files.praat

# This Praat script will change the sample rate of all sound files in a given directory
# and save AIFF files with the new rate to another directory.
# (See the Praat manual for details on resampling.)
# 
# This script is distributed under the GNU General Public License.
# Copyright 29.10.2003 Mietta Lennes

part$ = "1"
input_directory$ = "../DS_10283_2353/SiwisFrenchSpeechSynthesisDatabase/SiwisFrenchSpeechSynthesisDatabase/wavs/part"
output_directory$ = "../DS_10283_2353/SiwisFrenchSpeechSynthesisDatabase/SiwisFrenchSpeechSynthesisDatabase/rwavs/part"
new_sample_rate = 16000
precision = 50

fileList = do("Create Strings as file list...", "list", input_directory$ + part$ + "/*.wav")
numberOfFiles = do("Get number of strings")

for i to numberOfFiles

	selectObject(fileList)
	filename$ = do$("Get string...", i)
	soundObject =  do("Read from file...", input_directory$ + part$ + "/" + filename$)

	oldrate = Get sample rate
	if oldrate <> new_sample_rate
		printline Resampling 'input_directory$''part$'/'filename$' to 'new_sample_rate' Hz...
		Resample... new_sample_rate precision
	else
		printline Sample rate of 'input_directory$''part$'/'filename$' is already 'new_sample_rate' Hz, copying this file...
	endif

	Write to WAV file... 'output_directory$''part$'/'filename$'

	removeObject(soundObject)
endfor