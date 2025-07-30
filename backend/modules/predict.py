import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import os
from torch import nn
from .Transformer import Transformer

# --- Constants ---
PAD_TOKEN = 1
SOS_TOKEN = 2
EOS_TOKEN = 3
tokenize = lambda x: list(x)

# --- Load Vocab & Model ---
root = os.path.dirname(__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"

protein_vocab = torch.load(os.path.join(root, '../dl_models/protein-vocab.pt'),weights_only=False)
smiles_vocab = torch.load(os.path.join(root, '../dl_models/smiles-vocab.pt'),weights_only=False)

model = Transformer(
    src_tokens=len(protein_vocab),
    trg_tokens=len(smiles_vocab),
    dim_model=256,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dropout_p=0.1
).to(device)

model.load_state_dict(torch.load(os.path.join(root, '../checkpoints/checkpoint.pth'), map_location=device))
model.eval()

# --- Helpers ---
def protein_to_numbers(protein):
    return [protein_vocab[token] for token in tokenize(protein)]

def smiles_to_string(smiles):
    return ''.join([smiles_vocab.get_itos()[word] for word in smiles])

def generate_smiles(input_sequence, max_length=150):
    y_input = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)

    for _ in range(max_length):
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        pred = model(input_sequence, y_input, tgt_mask)
        next_item = pred.topk(1)[1].view(-1)[-1].item()
        next_item = torch.tensor([[next_item]], device=device)
        y_input = torch.cat((y_input, next_item), dim=1)

        if next_item.item() in [EOS_TOKEN, PAD_TOKEN]:
            break

    return y_input.view(-1).tolist()

def predict_smiles(protein_sequence):
    input_tensor = torch.tensor([protein_to_numbers(protein_sequence)], dtype=torch.long, device=device)
    result = generate_smiles(input_tensor)
    return smiles_to_string(result[1:-1])  # remove SOS and EOS
