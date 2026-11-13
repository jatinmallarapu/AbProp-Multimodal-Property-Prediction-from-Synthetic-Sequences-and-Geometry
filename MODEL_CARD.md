# Model card

The PyTorch fusion MLP accepts 16 positional tokens from an eight-symbol integer alphabet and three statistics of artificial point distances. It predicts three synthetic regression targets. The input geometry is normalized using training data. The benchmark holds out 12 generated families and uses a separate 12-family validation set. Results measure recovery of the chosen simulator rules. Intended use is demonstration of multimodal regression, group isolation, and ablation methodology.
