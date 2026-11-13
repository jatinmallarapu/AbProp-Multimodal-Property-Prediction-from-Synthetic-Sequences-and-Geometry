# AbProp

Multitask regression using artificial token sequences and spatial geometry. The generator creates 80 families, with 16 examples per family. Entire families are assigned to training, validation, or test. The three targets measure token signal, geometry signal, and their interaction. Inputs and labels are generated from numerical rules.

Run `python -m pip install -r requirements.txt`, then `python benchmark.py`. The run writes per-target RMSE and Spearman correlation, a predicted-versus-actual SVG, and fusion model weights under `results/`. Compare ridge regression on positional one-hot features against sequence-only, geometry-only, and fusion MLPs. Checkpoints are selected using validation loss. See [model card](MODEL_CARD.md).
