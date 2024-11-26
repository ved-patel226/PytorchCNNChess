import sys

sys.path.append("/mnt/Fedora2/code/python/tensorflow/rl")

from torchviz import make_dot
from CHESS import ChessPolicy
import torch

model = ChessPolicy()
checkpoint = torch.load("chess_policy_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dummy_input = torch.randn(1, 19, 8, 8)
output = model(dummy_input)

dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("chess_policy_model", format="png")
