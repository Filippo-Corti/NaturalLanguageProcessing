import torch

START_TOKEN = "<START>"
PADDING_TOKEN = "<PADDING>"
END_TOKEN = "<END>"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")