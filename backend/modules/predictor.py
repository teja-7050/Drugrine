import torch.nn as nn

import numpy as np
class QSAR(nn.Module):
    def __init__(self, params): 
        super(QSAR, self).__init__()
        self.params=params
        self.build_model(params)
    def build_model(self,params):
        if params["embedding"]=="basic_embedding" :
            embedding = nn.Embedding(num_embeddings=params["embedding_params"]["num_embeddings"],
                                      embedding_dim=params["embedding_params"]["embedding_dim"],
                                      padding_idx=params["embedding_params"]["padding_idx"])
        elif params["embedding"] == "finger_prints":
            embedding = None
        if params['encoder']=="RNNEncoder":
            
            self.encoder=self._init_rnn(params["encoder_params"])
        if params["mlp"]=="mlp":
            mlp_layers=[]
            mlp_dropout_layers=[]
            self.activation = params["mlp_params"]["activation"]
            if type(self.activation) is not list:
                self.activation = [self.activation] * params["mlp_params"]["n_layers"]
            else:
                assert len(self.activation) == params["mlp_params"]["n_layers"]
            mlp_activations=[]
            for n in range (params["mlp_params"]["n_layers"]):
                mlp_layer,dropout=self._init_mlp(params["mlp_params"],n)
                mlp_layers.append(mlp_layer)
                mlp_dropout_layers.append(dropout)
                mlp_activations.append(self.activation[n])
        
        self.embedding = embedding
        self.mlp = nn.ModuleList(mlp_layers)
        self.dropout=nn.ModuleList(mlp_dropout_layers)
        

        
    def _init_rnn(self,p):
        
        if p["layer"]=="LSTM":
            return nn.LSTM(p["input_size"],
                           p["encoder_dim"],
                            p["n_layers"],
                           p["dropout"],
                       batch_first=True)
        elif  p["layer"]=="RNN":
            return nn.RNN(p["input_size"],
                           p["encoder_dim"],
                            p["n_layers"],
                           p["dropout"],
                       batch_first=True)
        elif  p["layer"]=="GRU":
            return nn.GRU(input_size=self.params["encoder_params"]["input_size"],
                              hidden_size=self.params["encoder_params"]["encoder_dim"],
                              num_layers=self.params["encoder_params"]["n_layers"],
                              dropout=self.params["encoder_params"]["dropout"],
                              batch_first=True)

    def _init_mlp(self,p,n):
        input_size=p["input_size"]
        hidden_size=p["hidden_size"]
        input_size = [p['input_size']] + p["hidden_size"][:-1]
        
        dropout=nn.Dropout(p["dropout"])
        layer=nn.Linear(in_features=input_size[n], out_features=hidden_size[n])
        return layer,dropout
    def forward(self,inp):
        if self.embedding !=None:
            embedded = self.embedding(inp)
            out = embedded
        else: out=inp
        
    
        out, _ = self.encoder(out)  
        x=out
        for i, layer in enumerate(self.mlp):
            x = self.dropout[i](x)
            x = layer(x)
            x = self.activation[i](x)  
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
    def load_model(self, path, map_location=None):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {path}")
        
    def predict(self, smiles_list):
        """
        Predicts outputs for a list of SMILES strings.

        Args:
            smiles_list (list of str): List of SMILES strings.
            get_features (callable): Function to convert SMILES to input features (e.g., Morgan FP tensor).

        Returns:
            Tuple of (valid_smiles, predictions_tensor, invalid_smiles)
        """
        valid_smiles = []
        invalid_smiles = []

        for smi in smiles_list:
            if Chem.MolFromSmiles(smi):
                valid_smiles.append(smi)
            else:
                invalid_smiles.append(smi)

        if not valid_smiles:
            return [], torch.empty(0), invalid_smiles

        features = process_smiles_to_morgan_tensor(valid_smiles)
        self.eval()
        predictions=[]
        with torch.no_grad():
            for feature in features:
                predictions.append(self.forward(features))
        return valid_smiles, predictions, invalid_smiles
        
        

            
tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n',' ']
def get_tokens(smiles, tokens=None):
    """
    Returns list of unique tokens, token-2-index dictionary and
    number of unique tokens from the list of SMILES
    Args:
        smiles (list): list of SMILES strings to tokenize.
        tokens (string): string of tokens or None.
        If none will be extracted from dataset.
    Returns:
        tokens (list): list of unique tokens/SMILES alphabet.
        token2idx (dict): dictionary mapping token to its index.
        num_tokens (int): number of unique tokens.
    """
    if tokens is None:
        tokens = list(set(''.join(smiles)))
        tokens = sorted(tokens)
        tokens = ''.join(tokens)
    token2idx = dict((token, i) for i, token in enumerate(tokens))
    num_tokens = len(tokens)
    return tokens, token2idx, num_tokens

def seq2tensor(seqs, tokens, flip=True):
    tensor = np.zeros((len(seqs), len(seqs[0])))
    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            if seqs[i][j] in tokens:
                tensor[i, j] = tokens.index(seqs[i][j])
            else:
                tokens = tokens + seqs[i][j]
                tensor[i, j] = tokens.index(seqs[i][j])
    if flip:
        tensor = np.flip(tensor, axis=1).copy()
    return tensor, tokens
def pad_sequences(seqs, max_length=None, pad_symbol=' '):
    if max_length is None:
        max_length = -1
        for seq in seqs:
            max_length = max(max_length, len(seq))
    lengths = []
    for i in range(len(seqs)):
        cur_len = len(seqs[i])
        lengths.append(cur_len)
        seqs[i] = seqs[i] + pad_symbol * (max_length - cur_len)
    return seqs, lengths
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

import torch

def process_smiles_to_morgan_tensor(smiles_list, radius=2, nBits=2048):
    """
    Converts a list of SMILES strings to a PyTorch tensor of Morgan Fingerprints.

    Args:
        smiles_list (list of str): List of SMILES strings.
        radius (int): Radius for Morgan fingerprint.
        nBits (int): Number of bits for Morgan fingerprint.

    Returns:
        torch.Tensor: Tensor of Morgan Fingerprints [batch_size, nBits].
                     Returns an empty tensor if no valid SMILES are processed.
    """
    fingerprints = []
    generator = GetMorganGenerator(radius=radius, fpSize=nBits)
    for smiles in smiles_list:
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = generator.GetFingerprint(mol)
            fingerprints.append(list(map(int, fp)))  # Convert BitVect to list of ints
        
    if fingerprints:
        return torch.tensor(fingerprints, dtype=torch.float32)
    else:
        return torch.empty(0, nBits, dtype=torch.float32)