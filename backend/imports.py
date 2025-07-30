import re
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import shutil
import nibabel as nib
import cv2 

import joblib
from fastapi.responses import JSONResponse
from rdkit.Chem import Draw
from io import BytesIO
import tensorflow as tf
from fastapi.templating import Jinja2Templates
from patchify import patchify
import keras.backend as K
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
import uuid
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import io
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors,AllChem
from rdkit.Chem.Draw import MolToImage 
import base64
from IPython.display import display 
import torch
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import torch.nn.functional as F
import nibabel as nib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pathlib import Path
import os
import torch
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Query, APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
import os
import tempfile  # Import tempfile
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from modules.vit import load_model_vit,morgan_to_image
from modules.stackRNN import StackAugmentedRNN
from modules.data import GeneratorData
from modules.predictor import QSAR
from modules.chembert import RoBERTaForMaskedLM,chemberta_predict
from helper import identity,dice_coef,dice_loss
from modules.docking import download_pdb_and_ligand, convert_ligand_to_pdbqt, run_plif_and_visualize,add_hydrogens_to_protein
from modules.predict import predict_smiles