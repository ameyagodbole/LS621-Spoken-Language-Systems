# STEP 1
# Task Grammar

# Define the task grammar (see task_language/task_grammar)
# Here: 7 digits

# Generate word net from task specific grammar
HParse task_language/task_grammar task_language/wdnet

# -------------------
# -------------------

# Step 2
# Dictionary

# Define the pronunciation dictionary (see task_language/mr_pronunciation_dict)

# -------------------
# -------------------

# Step 3
# Recording the Data

# Given digit sequences in digitSequences40b.txt are converted to
# corresponding Marathi prompts using the python file en2mr.py
python en2mr.py prompts/digitSequences40b.txt task_language/en2mr_dict prompts/prompts.txt

# Generate extra test_prompts for further evaluation
# NOTE: Minor post-processing is required to change sample numbering
# HSGen -l -n 20 task_language/wdnet task_language/mr_pronunciation_dict > prompts/testprompts.txt
# python promt_file_pp.py prompts/testprompts.txt

# Record sentences using record_audio.py
# python record_audio.py prompts/prompts.txt      # Do not pass file name prefix
# python record_audio.py prompts/testprompts.txt  # Pass file name prefix 'test'

# -------------------
# -------------------

# Step 4
# Creating the Transcription Files

# Combine all prompts
cat prompts/prompts.txt prompts/testprompts.txt > prompts/allprompts.txt

# Generate word level MLF
perl prompts2mlf.pl label_files/allprompts.mlf prompts/allprompts.txt

# Generate phone level MLF
# without sp
HLEd -l '*' -d task_language/mr_pronunciation_dict -i label_files/phones0.mlf configs/mkphones0.led label_files/allprompts.mlf
# with sp
HLEd -l '*' -d task_language/mr_pronunciation_dict -i label_files/phones1.mlf configs/mkphones1.led label_files/allprompts.mlf

# Get required phones
python get_phones.py task_language/mr_pronunciation_dict task_language/monophones

# -------------------
# -------------------

# Step 5
# Coding the Data

# Create HCopy config file (HCopy_config_train and HCopy_config_test) for determining coding parameters
# Create script of HCopy wav -> mfc mapping
python create_code_scp.py scripts/codetr.scp wav sample mfcc
python create_code_scp.py scripts/codete.scp wav test mfcc

HCopy -A -D -T 1 -C configs/code_config -S scripts/codetr.scp
HCopy -A -D -T 1 -C configs/code_config -S scripts/codete.scp

# EXP1
# Train: on sample files of speaker 0
# Test: on test files of speaker 0 and speaker 1 and speaker 2
    # -------------------
    # -------------------

    # Step 6
    # Creating Flat Start Monophones

    # Create train script files
    python create_train_scp.py scripts/train_exp1.scp mfccMR_0 sample

    # Create hmm0 using global means and variances
    mkdir -p exp1/hmm0
    HCompV -C configs/hmm_config -f 0.01 -m -S scripts/train_exp1.scp -M exp1/hmm0 configs/proto0

    # Get hmm0 macros
    python create_hmm0_macros.py exp1/hmm0/macros exp1/hmm0/vFloors

    # Get hmmdefs
    python create_hmm0_defs.py exp1/hmm0/hmmdefs exp1/hmm0/proto0 task_language/monophones0

    # Re-estimation run 1
    # Create hmm1
    mkdir exp1/hmm1
    HERest -C configs/hmm_config -I label_files/phones0.mlf -t 250.0 150.0 1000.0 \
    -S scripts/train_exp1.scp -H exp1/hmm0/macros -H exp1/hmm0/hmmdefs -M exp1/hmm1 task_language/monophones0

    # Re-estimation run 2
    # Create hmm2
    mkdir exp1/hmm2
    HERest -C configs/hmm_config -I label_files/phones0.mlf -t 250.0 150.0 1000.0 \
    -S scripts/train_exp1.scp -H exp1/hmm1/macros -H exp1/hmm1/hmmdefs -M exp1/hmm2 task_language/monophones0

    # Re-estimation run 3
    # Create hmm3
    mkdir exp1/hmm3
    HERest -C configs/hmm_config -I label_files/phones0.mlf -t 250.0 150.0 1000.0 \
    -S scripts/train_exp1.scp -H exp1/hmm2/macros -H exp1/hmm2/hmmdefs -M exp1/hmm3 task_language/monophones0

    # -------------------
    # -------------------

    # Step 7
    # Fixing the Silence Models

    # Copy hmm3 to hmm4
    mkdir exp1/hmm4
    cp -R exp1/hmm3/* exp1/hmm4/

    # Create sp model in hmm4/hmmdefs
    python create_sp_model.py exp1/hmm3/hmmdefs exp1/hmm4/hmmdefs

    # Create hmm5 (tie state 3 of sil and state 2 of sp) (add transitions in sil and sp)
    mkdir exp1/hmm5
    HHEd -H exp1/hmm4/macros -H exp1/hmm4/hmmdefs -M exp1/hmm5 configs/sil.hed task_language/monophones1

    # Re-estimation run 1
    # Create hmm6
    mkdir exp1/hmm6
    HERest -C configs/hmm_config -I label_files/phones1.mlf -t 250.0 150.0 1000.0 \
    -S scripts/train_exp1.scp -H exp1/hmm5/macros -H exp1/hmm5/hmmdefs -M exp1/hmm6 task_language/monophones1

    # Re-estimation run 2
    # Create hmm7
    mkdir exp1/hmm7
    HERest -C configs/hmm_config -I label_files/phones1.mlf -t 250.0 150.0 1000.0 \
    -S scripts/train_exp1.scp -H exp1/hmm6/macros -H exp1/hmm6/hmmdefs -M exp1/hmm7 task_language/monophones1

    # -------------------
    # -------------------

    # Step 8
    # Realigning the Training Data

    # NOTE: Although the given dictionary does not contain words with multiple pronunciations,
    # this step is performed for completeness

    HVite -l '*' -o SWT -b SILENCE -C configs/hmm_config -a -H exp1/hmm7/macros \
        -H exp1/hmm7/hmmdefs -i label_files/aligned_exp1.mlf -m -t 250.0 -y lab \
        -I label_files/allprompts.mlf -S scripts/train_exp1.scp task_language/mr_pronunciation_dict task_language/monophones1

    # Re-estimation run 1
    # Create hmm8
    mkdir exp1/hmm8
    HERest -C configs/hmm_config -I label_files/aligned_exp1.mlf -t 250.0 150.0 1000.0 \
    -S scripts/train_exp1.scp -H exp1/hmm7/macros -H exp1/hmm7/hmmdefs -M exp1/hmm8 task_language/monophones1

    # Re-estimation run 2
    # Create hmm9
    mkdir exp1/hmm9
    HERest -C configs/hmm_config -I label_files/aligned_exp1.mlf -t 250.0 150.0 1000.0 \
    -S scripts/train_exp1.scp -H exp1/hmm8/macros -H exp1/hmm8/hmmdefs -M exp1/hmm9 task_language/monophones1

    # -------------------
    # -------------------

    # Step 9
    # Test HMMs

    # Create test script files
    python create_train_scp.py scripts/test1.scp mfccMR_0 test
    python create_train_scp.py scripts/test2.scp mfccMR_1 test
    python create_train_scp.py scripts/test3.scp mfccMR_2 test

    # Test 1
    HVite -H exp1/hmm9/macros -H exp1/hmm9/hmmdefs -S scripts/test1.scp \
    -l '*' -i results/recout_test1_exp1.mlf -w task_language/wdnet \
    -p 0.0 -s 5.0 task_language/mr_pronunciation_dict task_language/monophones1

    HResults -I label_files/allprompts.mlf task_language/monophones1 results/recout_test1_exp1.mlf > results/stats/exp1_test1.txt

    # Test 2
    HVite -H exp1/hmm9/macros -H exp1/hmm9/hmmdefs -S scripts/test2.scp \
    -l '*' -i results/recout_test2_exp1.mlf -w task_language/wdnet \
    -p 0.0 -s 5.0 task_language/mr_pronunciation_dict task_language/monophones1

    HResults -I label_files/allprompts.mlf task_language/monophones1 results/recout_test2_exp1.mlf > results/stats/exp1_test2.txt

    # Test 3
    HVite -H exp1/hmm9/macros -H exp1/hmm9/hmmdefs -S scripts/test3.scp \
    -l '*' -i results/recout_test3_exp1.mlf -w task_language/wdnet \
    -p 0.0 -s 5.0 task_language/mr_pronunciation_dict task_language/monophones1

    HResults -I label_files/allprompts.mlf task_language/monophones1 results/recout_test3_exp1.mlf > results/stats/exp1_test3.txt

# EXP2
# Train: on sample files of speaker 0 and speaker 1
# Test: on test files of speaker 0 and speaker 1 and speaker 2
    # -------------------
    # -------------------

    # Step 6
    # Creating Flat Start Monophones

    # Create train script files
    python create_train_scp.py scripts/train_exp2.scp mfccMR_ sample

    # Create hmm0 using global means and variances
    mkdir -p exp2/hmm0
    HCompV -C configs/hmm_config -f 0.01 -m -S scripts/train_exp2.scp -M exp2/hmm0 configs/proto0

    # Get hmm0 macros
    python create_hmm0_macros.py exp2/hmm0/macros exp2/hmm0/vFloors

    # Get hmmdefs
    python create_hmm0_defs.py exp2/hmm0/hmmdefs exp2/hmm0/proto0 task_language/monophones0

    # Re-estimation run 1
    # Create hmm1
    mkdir exp2/hmm1
    HERest -C configs/hmm_config -I label_files/phones0.mlf -t 250.0 150.0 1000.0 \
    -S scripts/train_exp2.scp -H exp2/hmm0/macros -H exp2/hmm0/hmmdefs -M exp2/hmm1 task_language/monophones0

    # Re-estimation run 2
    # Create hmm2
    mkdir exp2/hmm2
    HERest -C configs/hmm_config -I label_files/phones0.mlf -t 250.0 150.0 1000.0 \
    -S scripts/train_exp2.scp -H exp2/hmm1/macros -H exp2/hmm1/hmmdefs -M exp2/hmm2 task_language/monophones0

    # Re-estimation run 3
    # Create hmm3
    mkdir exp2/hmm3
    HERest -C configs/hmm_config -I label_files/phones0.mlf -t 250.0 150.0 1000.0 \
    -S scripts/train_exp2.scp -H exp2/hmm2/macros -H exp2/hmm2/hmmdefs -M exp2/hmm3 task_language/monophones0

    # -------------------
    # -------------------

    # Step 7
    # Fixing the Silence Models

    # Copy hmm3 to hmm4
    mkdir exp2/hmm4
    cp -R exp2/hmm3/* exp2/hmm4/

    # Create sp model in hmm4/hmmdefs
    python create_sp_model.py exp2/hmm3/hmmdefs exp2/hmm4/hmmdefs

    # Create hmm5 (tie state 3 of sil and state 2 of sp) (add transitions in sil and sp)
    mkdir exp2/hmm5
    HHEd -H exp2/hmm4/macros -H exp2/hmm4/hmmdefs -M exp2/hmm5 configs/sil.hed task_language/monophones1

    # Re-estimation run 1
    # Create hmm6
    mkdir exp2/hmm6
    HERest -C configs/hmm_config -I label_files/phones1.mlf -t 250.0 150.0 1000.0 \
    -S scripts/train_exp2.scp -H exp2/hmm5/macros -H exp2/hmm5/hmmdefs -M exp2/hmm6 task_language/monophones1

    # Re-estimation run 2
    # Create hmm7
    mkdir exp2/hmm7
    HERest -C configs/hmm_config -I label_files/phones1.mlf -t 250.0 150.0 1000.0 \
    -S scripts/train_exp2.scp -H exp2/hmm6/macros -H exp2/hmm6/hmmdefs -M exp2/hmm7 task_language/monophones1

    # -------------------
    # -------------------

    # Step 8
    # Realigning the Training Data

    # NOTE: Although the given dictionary does not contain words with multiple pronunciations,
    # this step is peerformed for completeness

    HVite -l '*' -o SWT -b SILENCE -C configs/hmm_config -a -H exp2/hmm7/macros \
        -H exp2/hmm7/hmmdefs -i label_files/aligned_exp2.mlf -m -t 250.0 -y lab \
        -I label_files/allprompts.mlf -S scripts/train_exp2.scp task_language/mr_pronunciation_dict task_language/monophones1

    # Re-estimation run 1
    # Create hmm8
    mkdir exp2/hmm8
    HERest -C configs/hmm_config -I label_files/aligned_exp2.mlf -t 250.0 150.0 1000.0 \
    -S scripts/train_exp2.scp -H exp2/hmm7/macros -H exp2/hmm7/hmmdefs -M exp2/hmm8 task_language/monophones1

    # Re-estimation run 2
    # Create hmm9
    mkdir exp2/hmm9
    HERest -C configs/hmm_config -I label_files/aligned_exp2.mlf -t 250.0 150.0 1000.0 \
    -S scripts/train_exp2.scp -H exp2/hmm8/macros -H exp2/hmm8/hmmdefs -M exp2/hmm9 task_language/monophones1

    # -------------------
    # -------------------

    # Step 9
    # Test HMMs

    # Create test script files
    python create_train_scp.py scripts/test1.scp mfccMR_0 test
    python create_train_scp.py scripts/test2.scp mfccMR_1 test

    # Test 1
    HVite -H exp2/hmm9/macros -H exp2/hmm9/hmmdefs -S scripts/test1.scp \
    -l '*' -i results/recout_test1_exp2.mlf -w task_language/wdnet \
    -p 0.0 -s 5.0 task_language/mr_pronunciation_dict task_language/monophones1

    HResults -I label_files/allprompts.mlf task_language/monophones1 results/recout_test1_exp2.mlf > results/stats/exp2_test1.txt

    # Test 2
    HVite -H exp2/hmm9/macros -H exp2/hmm9/hmmdefs -S scripts/test2.scp \
    -l '*' -i results/recout_test2_exp2.mlf -w task_language/wdnet \
    -p 0.0 -s 5.0 task_language/mr_pronunciation_dict task_language/monophones1

    HResults -I label_files/allprompts.mlf task_language/monophones1 results/recout_test2_exp2.mlf > results/stats/exp2_test2.txt

    # Test 3
    HVite -H exp2/hmm9/macros -H exp2/hmm9/hmmdefs -S scripts/test3.scp \
    -l '*' -i results/recout_test3_exp2.mlf -w task_language/wdnet \
    -p 0.0 -s 5.0 task_language/mr_pronunciation_dict task_language/monophones1

    HResults -I label_files/allprompts.mlf task_language/monophones1 results/recout_test3_exp2.mlf > results/stats/exp2_test3.txt