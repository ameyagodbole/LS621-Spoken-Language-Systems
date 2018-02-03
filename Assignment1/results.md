1.	Switching from raw FFT power spectrum to ceptral coefficients solves all classification errors (without Cepstral Mean Normalization (CMN)).

2.	Here, it is observed that CMN actually degrades performance. A possible reason for this is that although CMN works well for achieving speaker independence, it is not robust to additive noise.

3.	DTW using constraints on the warping function is implemented and demonstrated.

4.	DTW achieves 100% accuracy for all words.

5.	For completeness, creation of composite template warped by DTW is also performed.
