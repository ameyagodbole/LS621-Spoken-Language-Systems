#### BUILDING COMPOSITE REFERENCE:

1.	The function dtw in dtw.py returns the cost as well as the path of the optimal warping of reference template and input. This path maps each frame in the input utterance to a frame in the reference utterance.

2.	By performing DTW between the 2 reference utterances, we get a mapping between the frames. Without loss of generality, I have chosen the longer utterance to act as reference 1 and the shorter one as reference 2.

3.	Given the mapping M, the algorithm for genrating the composite reference is:
NOTE:	reference(i) := MFCC coefficients of frame i

```
for i in range(len(reference1)):

	count = 1.
	for each frame j of reference2 mapping to a frame i of reference1 in M:
		reference1(i) = reference1(i) + reference2(j)
		count++;
	end for

	reference1(i) /= count
end for
```

reference1 will thus contain the generated composite.


#### RESULTS:

*	Switching from raw FFT power spectrum to ceptral coefficients solves all classification errors (without Cepstral Mean Normalization (CMN)).

*	Here, it is observed that CMN actually degrades performance. A possible reason for this is that although CMN works well for achieving speaker independence, it is not robust to additive noise.

*	DTW using constraints on the warping function is implemented and demonstrated.

*	DTW achieves 100% accuracy for all words.

*	For completeness, creation of composite template warped by DTW is also performed.
