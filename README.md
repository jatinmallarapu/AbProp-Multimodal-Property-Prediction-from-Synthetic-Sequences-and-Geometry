# AbProp

Multitask regression using artificial token sequences and spatial geometry. The generator creates 80 families, with 16 examples per family. Entire families are assigned to training, validation, or test. The three targets measure token signal, geometry signal, and their interaction. Inputs and labels are generated from numerical rules.

Run `python -m pip install -r requirements.txt`, then `python benchmark.py`. The run writes per-target RMSE and Spearman correlation, a predicted-versus-actual SVG, and fusion model weights under `results/`. Compare ridge regression on positional one-hot features against sequence-only, geometry-only, and fusion MLPs. Checkpoints are selected using validation loss. See [model card](MODEL_CARD.md).

The workflow artifact includes the fusion checkpoint with geometry and target normalization.

## Held-out results

| Model | Token RMSE / Spearman | Geometry RMSE / Spearman | Interaction RMSE / Spearman |
| --- | ---: | ---: | ---: |
| ridge | 0.037 / 0.996 | 0.445 / -0.017 | 0.297 / 0.011 |
| sequence | 0.163 / 0.976 | 0.406 / 0.036 | 0.251 / 0.137 |
| geometry | 0.605 / -0.006 | 0.037 / 0.995 | 0.268 / -0.078 |
| fusion | 0.056 / 0.994 | 0.045 / 0.994 | 0.074 / 0.949 |

Fusion reduced interaction RMSE from 0.251 for the best single-modality model to 0.074. Ridge had the lowest token RMSE and the geometry-only MLP had the lowest geometry RMSE. The 192 test examples come from 12 held-out families and the results are for seed 42.

[Verified run](https://github.com/jatinmallarapu/AbProp-Multimodal-Property-Prediction-from-Synthetic-Sequences-and-Geometry/actions/runs/43386608514) · [Results, plot and checkpoint](https://github.com/jatinmallarapu/AbProp-Multimodal-Property-Prediction-from-Synthetic-Sequences-and-Geometry/actions/runs/43386883809/artifacts/14018604043)
