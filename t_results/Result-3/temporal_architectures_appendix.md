# Appendix: Full hybrid_ts configuration sweep (48 runs)

This appendix accompanies *NFL Tackle Prediction: From Single-Frame to True Temporal Modeling* and lists every hybrid_ts configuration from the 192-run temporal sweep (model_dim in {32, 128, 512} x num_layers in {1, 2, 4, 8} x window T in {5, 10, 15, 20} = 48 runs). All use the standard recipe (AdamW, lr 1e-4, batch 256, dropout 0.3, SmoothL1Loss, patience 10, max 200 epochs, 8 raw features). Models are selected by validation loss; test ADE is reported but not used for selection. Window T is in frames (10 frames = 1 second). FLOPs are per single inference. Numbers are from t_results/model_comparison.json.

The selected best configuration is **M128 / L4 / W15** (lowest validation loss, 2.256), which is also the lowest test ADE (4.09); it is shown in bold.

| model_dim | num_layers | window T (frames) | params | inference FLOPs | val loss | test ADE (yd) |
|---|---|---|---|---|---|---|
| 32 | 1 | 5 | 18,106 | 1,422,288 | 2.597 | 4.67 |
| 32 | 1 | 10 | 18,106 | 2,293,488 | 2.630 | 4.74 |
| 32 | 1 | 15 | 18,106 | 3,164,688 | 2.711 | 4.86 |
| 32 | 1 | 20 | 18,106 | 4,035,888 | 2.544 | 4.60 |
| 32 | 2 | 5 | 30,810 | 1,970,000 | 2.545 | 4.61 |
| 32 | 2 | 10 | 30,810 | 2,841,200 | 2.544 | 4.59 |
| 32 | 2 | 15 | 30,810 | 3,712,400 | 2.478 | 4.51 |
| 32 | 2 | 20 | 30,810 | 4,583,600 | 2.468 | 4.49 |
| 32 | 4 | 5 | 56,218 | 3,065,424 | 2.431 | 4.42 |
| 32 | 4 | 10 | 56,218 | 3,936,624 | 2.360 | 4.32 |
| 32 | 4 | 15 | 56,218 | 4,807,824 | 2.419 | 4.39 |
| 32 | 4 | 20 | 56,218 | 5,679,024 | 2.523 | 4.59 |
| 32 | 8 | 5 | 107,034 | 5,256,272 | 2.390 | 4.36 |
| 32 | 8 | 10 | 107,034 | 6,127,472 | 2.383 | 4.34 |
| 32 | 8 | 15 | 107,034 | 6,998,672 | 2.348 | 4.29 |
| 32 | 8 | 20 | 107,034 | 7,869,872 | 2.365 | 4.28 |
| 128 | 1 | 5 | 272,050 | 20,312,736 | 2.347 | 4.29 |
| 128 | 1 | 10 | 272,050 | 31,902,336 | 2.350 | 4.29 |
| 128 | 1 | 15 | 272,050 | 43,491,936 | 2.347 | 4.33 |
| 128 | 1 | 20 | 272,050 | 55,081,536 | 2.362 | 4.33 |
| 128 | 2 | 5 | 470,322 | 28,991,648 | 2.312 | 4.26 |
| 128 | 2 | 10 | 470,322 | 40,581,248 | 2.310 | 4.22 |
| 128 | 2 | 15 | 470,322 | 52,170,848 | 2.303 | 4.24 |
| 128 | 2 | 20 | 470,322 | 63,760,448 | 2.287 | 4.23 |
| 128 | 4 | 5 | 866,866 | 46,349,472 | 2.268 | 4.13 |
| 128 | 4 | 10 | 866,866 | 57,939,072 | 2.270 | 4.12 |
| **128** | **4** | **15** | **866,866** | **69,528,672** | **2.256** | **4.09** |
| 128 | 4 | 20 | 866,866 | 81,118,272 | 2.268 | 4.15 |
| 128 | 8 | 5 | 1,659,954 | 81,065,120 | 2.271 | 4.15 |
| 128 | 8 | 10 | 1,659,954 | 92,654,720 | 2.284 | 4.12 |
| 128 | 8 | 15 | 1,659,954 | 104,244,320 | 2.290 | 4.12 |
| 128 | 8 | 20 | 1,659,954 | 115,833,920 | 2.304 | 4.19 |
| 512 | 1 | 5 | 4,283,026 | 315,307,488 | 2.399 | 4.38 |
| 512 | 1 | 10 | 4,283,026 | 491,421,888 | 2.364 | 4.31 |
| 512 | 1 | 15 | 4,283,026 | 667,536,288 | 2.349 | 4.31 |
| 512 | 1 | 20 | 4,283,026 | 843,650,688 | 2.364 | 4.31 |
| 512 | 2 | 5 | 7,435,410 | 453,832,160 | 2.362 | 4.25 |
| 512 | 2 | 10 | 7,435,410 | 629,946,560 | 2.350 | 4.30 |
| 512 | 2 | 15 | 7,435,410 | 806,060,960 | 2.346 | 4.28 |
| 512 | 2 | 20 | 7,435,410 | 982,175,360 | 2.340 | 4.23 |
| 512 | 4 | 5 | 13,740,178 | 730,881,504 | 2.334 | 4.24 |
| 512 | 4 | 10 | 13,740,178 | 906,995,904 | 2.334 | 4.19 |
| 512 | 4 | 15 | 13,740,178 | 1,083,110,304 | 2.333 | 4.22 |
| 512 | 4 | 20 | 13,740,178 | 1,259,224,704 | 2.317 | 4.19 |
| 512 | 8 | 5 | 26,349,714 | 1,284,980,192 | 2.335 | 4.24 |
| 512 | 8 | 10 | 26,349,714 | 1,461,094,592 | 2.331 | 4.20 |
| 512 | 8 | 15 | 26,349,714 | 1,637,208,992 | 2.352 | 4.24 |
| 512 | 8 | 20 | 26,349,714 | 1,813,323,392 | 2.337 | 4.20 |

A few patterns visible in the full sweep:

- **Width:** the jump from M32 to M128 is the large one (best test ADE 4.28 at M32, 4.09 at M128). M512 does not improve on M128 (best 4.19) despite roughly 16 to 30 times the parameters and up to 26 times the FLOPs, so capacity beyond M128 is wasted on this task.
- **Depth:** within M128, 4 layers is the sweet spot; 1 layer underfits (best 4.29) and 8 layers does not beat 4 (best 4.12).
- **Window:** within M128 / L4, test ADE is nearly flat across T (4.13, 4.12, 4.09, 4.15 for T = 5, 10, 15, 20), confirming history plateaus around 1.0 to 1.5 seconds.
- **Selection by validation loss is well behaved:** the lowest val loss (2.256) coincides with the lowest test ADE (4.09), so the selected model is not an artifact of the selection metric.
