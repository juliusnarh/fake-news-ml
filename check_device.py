import torch
print("torch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

import shap, numpy as np
print("shap version:", shap.__version__)