# EDITED
# source: Baidu Deep Voice utility
# https://github.com/baidu-research/deep-voice/blob/master/scripts/f0-script.praat

timeStep = 0.005
minimum_pitch = 75
maximum_pitch = 500
part$ = "3"
input_directory$ = "../DS_10283_2353/SiwisFrenchSpeechSynthesisDatabase/SiwisFrenchSpeechSynthesisDatabase/rwavs/part"
output_directory$ = "../DS_10283_2353/SiwisFrenchSpeechSynthesisDatabase/SiwisFrenchSpeechSynthesisDatabase/f0/part"
file_type$ = "wav"

fileList = do("Create Strings as file list...", "list", input_directory$ + part$ + "/*." + file_type$)
numberOfFiles = do("Get number of strings")

for i to numberOfFiles

  selectObject(fileList)
  filename$ = do$("Get string...", i)
  soundObject =  do("Read from file...", input_directory$ + part$ + "/" + filename$)

  pitchObject = do("To Pitch...", timeStep, minimum_pitch, maximum_pitch)
  removeObject(soundObject)
  tableObject = do("Create Table with column names...", "table", 0, 
      ..."time pitch")

  selectObject(pitchObject)
  numberOfFrames = do("Get number of frames")
  for frame to numberOfFrames

    select pitchObject
    f0 = do("Get value in frame...", frame, "Hertz")
    time = do("Get time from frame number...", frame)

    selectObject(tableObject)
    do("Append row")
    thisRow = do("Get number of rows")
    do("Set numeric value...", thisRow, "time", time)
    do("Set numeric value...", thisRow, "pitch", f0)

  endfor

  filename$ = filename$ - ("." + file_type$)
  do("Write to table file...", output_directory$ + part$ + "/" + filename$ + ".txt")
  removeObject(tableObject, pitchObject)
endfor