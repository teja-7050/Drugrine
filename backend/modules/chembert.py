import torch
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F





class RoBERTaEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, max_len=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        positions = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        x = self.token_embed(input_ids) + self.position_embed(positions)
        return self.dropout(self.norm(x))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, D = x.size()
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        return self.dropout(self.out_proj(out))

class CustomTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden=512):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x
    

class RoBERTaForMaskedLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_layers=4, max_len=128):
        super().__init__()
        self.embedding = RoBERTaEmbedding(vocab_size, embed_dim, max_len)
        self.encoder = nn.Sequential(*[
            CustomTransformerBlock(embed_dim, num_heads=8)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        return self.lm_head(x)

class SMILESDataset(Dataset):
    def __init__(self, smiles, tokenizer, max_len=128):
        self.smiles = smiles
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.smiles[idx])
        input_ids = encoded.ids
        input_ids += [vocab["[PAD]"]] * (self.max_len - len(input_ids))
        return torch.tensor(input_ids[:self.max_len], dtype=torch.long)

def encode_input(smiles, tokenizer, vocab):
    tokens = []
    print(f"Original SMILES: {smiles}")  # Print the original SMILES
    encoded = tokenizer.encode(smiles)
    print(f"Encoded Tokens: {encoded.tokens}")  # Print the tokenized representation
    print(f"Encoded Token IDs: {encoded.ids}")  # Print the corresponding token IDs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for token in encoded.tokens:
        if token == "<mask>":
            tokens.append(vocab["<mask>"])
        else:
            tokens.append(vocab.get(token, vocab["[UNK]"]))
    return torch.tensor(tokens).unsqueeze(0).to(device)




def chemberta_predict(smiles_input,tokenizer, vocab,device,model):
    input_ids = encode_input(smiles_input, tokenizer, vocab)
    print(f"Input IDs: {input_ids}")  # Print the input IDs tensor

    mask_token_id = vocab["<mask>"]
    print(f"Mask Token ID: {mask_token_id}")  # Print the ID for <mask>

    mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]
    print(f"Mask Indices: {mask_indices}")  # Print the indices where mask is found

    if len(mask_indices) == 0:
        return "No <mask> token found in input!"

    mask_index = mask_indices[0].item()

    with torch.no_grad():
        logits = model(input_ids)
        mask_logits = logits[0, mask_index]

        # Convert to probabilities
        probs = F.softmax(mask_logits, dim=-1)

        # Get top 5 predictions
        topk_probs, topk_indices = torch.topk(probs, k=5)
        topk_probs = topk_probs.cpu().numpy()
        topk_indices = topk_indices.cpu().numpy()

        results = []
        for prob, token_id in zip(topk_probs, topk_indices):
            token = list(vocab.keys())[list(vocab.values()).index(token_id)]
            results.append(f"{token} ({prob:.4f})")

        # Combine top 5 into a single string for display
        return "Top 5 Predictions:\n" + "\n".join(results)
