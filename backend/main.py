from imports import * 

import logging
import traceback
import MDAnalysis as mda
import subprocess
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

"""

VIT MODEL

"""

vit, le = load_model_vit("dl_models/vit_model.pkl")
"""
CHEMBERT-A MODEL
_
"""

bpe_model = BPE.from_file("datasets/updated_vocab.json", "datasets/merges.txt")
tokenizer = Tokenizer(bpe_model)
tokenizer.add_special_tokens(['<mask>'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("datasets/updated_vocab.json", "r") as f:
    vocab = json.load(f)
chembert_model = RoBERTaForMaskedLM(len(vocab))
chembert_model.load_state_dict(torch.load("dl_models/model.pth", map_location=device))
chembert_model.to(device)
chembert_model.eval()

"""
QSAR MODEL nd STACK RNN MODEL

"""
n_hidden = 512
batch_size = 128
num_epochs = 50
lr = 0.005

model_params = {
    'embedding': "finger_prints",
    'embedding_params': {
        'embedding_dim': n_hidden,
        'fingerprint_dim': 2048  
    },
    'encoder': "RNNEncoder",
    'encoder_params': {
        'input_size': 2048,
        'layer': "GRU",
        'encoder_dim': n_hidden,
        'n_layers': 2,
        'dropout': 0.8
    },
    'mlp': "mlp",
    'mlp_params': {
        'input_size': n_hidden,
        'n_layers': 2,
        'hidden_size': [n_hidden, 1],
        'activation': [F.relu, identity],
        'dropout': 0.0
    }
}

gen_data_path = 'datasets/chembl_22_clean_1576904_sorted_std_final (1).smi'

tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
predictor_pic50=QSAR(model_params)
predictor_logP=QSAR(model_params)
predictor_pic50.load_model('dl_models/qsar_model_pic50.pt')
predictor_logP.load_model('dl_models/qsar_model_logP.pt')
gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', 
                         cols_to_read=[0], keep_header=True, tokens=tokens)
hidden_size = 1500
stack_width = 1500
stack_depth = 200
layer_type = 'GRU'
lr = 0.001
optimizer_instance = torch.optim.Adadelta
my_generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                 output_size=gen_data.n_characters, layer_type=layer_type,
                                 n_layers=1, is_bidirectional=False, has_stack=True,
                                 stack_width=stack_width, stack_depth=stack_depth, 
                                 use_cuda=None, 
                                 optimizer_instance=optimizer_instance, lr=lr)
my_generator.load_model('dl_models/latest')

my_generator_logp = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                 output_size=gen_data.n_characters, layer_type=layer_type,
                                 n_layers=1, is_bidirectional=False, has_stack=True,
                                 stack_width=stack_width, stack_depth=stack_depth, 
                                 use_cuda=None, 
                                 optimizer_instance=optimizer_instance, lr=lr)
my_generator_logp.load_model('dl_models/logPoptim')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER1 = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER1 = os.path.join(BASE_DIR, "../frontend/public")
MODEL_PATHS = {
    "brain": os.path.join(BASE_DIR, "dl_models", "model256.keras"),
    "lung": os.path.join(BASE_DIR, "dl_models", "lungmodel.keras")
}

os.makedirs(UPLOAD_FOLDER1, exist_ok=True)
os.makedirs(OUTPUT_FOLDER1, exist_ok=True)

# Load model function
def load_model(model_type: str):
    model_path = MODEL_PATHS.get(model_type)
    if not model_path or not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"

    try:
        model = tf.keras.models.load_model(model_path, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef})
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

# Process image function
async def process_image(file: UploadFile, model):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None, f"Unable to read image: {file.filename}"

        image = cv2.resize(image, (256, 256)) / 255.0
        patches = patchify(image, (16, 16, 3), 16).reshape(1, -1, 16 * 16 * 3)

        pred = model.predict(patches, verbose=0)[0]
        pred_mask = np.where(pred > 0.5, 255, 0).astype(np.uint8)  # Binary mask (0 or 255)
        pred_mask_resized = pred_mask.reshape((256, 256))

        # Create a color mask (e.g., red overlay)
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[:, :, 0] = pred_mask_resized  # Red channel

        # Overlay the color mask on the original image (optional, for visualization)
        overlay = cv2.addWeighted(image.astype(np.float32), 0.7, color_mask.astype(np.float32) / 255.0, 0.3, 0)
        overlay_bgr = (overlay * 255).astype(np.uint8)

        # Save the mask (you might want to save the overlay instead or both)
        mask_filename = f"mask_{uuid.uuid4().hex}_{file.filename}"
        mask_output_path = os.path.join(OUTPUT_FOLDER1, mask_filename)
        print("/" + mask_filename)
        if cv2.imwrite(mask_output_path, pred_mask_resized):
            return "/" + mask_filename, True # Return path relative to public folder

        return None, "Failed to save prediction mask"

    except Exception as e:
        return None, f"Error processing image: {e}"
    import requests
# Initialize Flask app and CORS
# app = Flask(_name_)
# #CORS(app)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

# Global cache dictionaries for storing input-output mappings
property_prediction_cache = {}
reinforcement_generation_cache = {}

# Cache file paths (optional: change as needed)
property_cache_file = "property_prediction_cache.pkl"
reinforcement_cache_file = "reinforcement_generation_cache.pkl"

# Load cached data from files (if exists)
def load_cache():
    global property_prediction_cache, reinforcement_generation_cache
    if Path(property_cache_file).exists():
        property_prediction_cache = joblib.load(property_cache_file)
    if Path(reinforcement_cache_file).exists():
        reinforcement_generation_cache = joblib.load(reinforcement_cache_file)

# Save cached data to files
def save_cache():
    joblib.dump(property_prediction_cache, property_cache_file)
    joblib.dump(reinforcement_generation_cache, reinforcement_cache_file)

# Load cache when the app starts
load_cache()


# Load Reinforcement Learning Model (StackAugmentedRNN)
# gen_data_path = 'C:\\Users\\jhans\\OneDrive\\Desktop\\ps-1 (2)\\ps-1\\backend\\flask\\123.smi'  # Adjust path if needed
tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', cols_to_read=[0], keep_header=True, tokens=tokens)

my_generator = StackAugmentedRNN(
    input_size=gen_data.n_characters,
    hidden_size=1500,
    output_size=gen_data.n_characters,
    layer_type='GRU',
    n_layers=1,
    is_bidirectional=False,
    has_stack=True,
    stack_width=1500,
    stack_depth=200,
    use_cuda=None,
    optimizer_instance=torch.optim.Adadelta,
    lr=0.001
)
# my_generator.load_model('C:\\Users\\jhans\\OneDrive\\Desktop\\ps-1 (2)\\ps-1\\backend\\flask\\latest')  # Make sure model is present

# --- Helper Functions ---
def smiles_to_fp_array(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(list(fp.ToBitString())).astype(int)

def calculate_molecular_properties(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    properties = {}
    if molecule is not None:
        properties['Molecular Weight'] = Descriptors.MolWt(molecule)
        properties['LogP'] = Descriptors.MolLogP(molecule)
        properties['H-Bond Donor Count'] = Descriptors.NumHDonors(molecule)
        properties['H-Bond Acceptor Count'] = Descriptors.NumHAcceptors(molecule)
        properties['pIC50'] = 5 + 0.1 * properties['LogP'] + 0.01 * properties['Molecular Weight']
    return properties

def smiles_to_base64(smiles):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(300, 300))
    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def compute_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)

# --- Flask Route ---

BASE_DIR = Path(__file__).resolve().parent
# No longer using FRONTEND_DIR
# FRONTEND_DIR = BASE_DIR.parent / "frontend"
# PUBLIC_FOLDER = FRONTEND_DIR / "public"
# ORIGINAL_NII_PUBLIC_FOLDER = PUBLIC_FOLDER / "original_nii"
# PREDICTED_NII_PUBLIC_FOLDER = PUBLIC_FOLDER / "predicted_nii"

# Define upload and processing directories
UPLOAD_FOLDER = BASE_DIR / "temp_nii_uploads"
SLICES_FOLDER = BASE_DIR / "temp_nii_slices"
OUTPUT_MASKS_FOLDER = BASE_DIR / "temp_nii_masks"
PUBLIC_FOLDER = BASE_DIR / "static"  # Define the public folder in the current directory
ORIGINAL_NII_PUBLIC_FOLDER = PUBLIC_FOLDER / "original_nii" #create original and predicted nii folders
PREDICTED_NII_PUBLIC_FOLDER = PUBLIC_FOLDER / "predicted_nii"


# Ensure output directories exist
# ORIGINAL_NII_PUBLIC_FOLDER.mkdir(parents=True, exist_ok=True) # changed
# PREDICTED_NII_PUBLIC_FOLDER.mkdir(parents=True, exist_ok=True) # changed
PUBLIC_FOLDER.mkdir(parents=True, exist_ok=True)
ORIGINAL_NII_PUBLIC_FOLDER.mkdir(parents=True, exist_ok=True)
PREDICTED_NII_PUBLIC_FOLDER.mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
SLICES_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_MASKS_FOLDER.mkdir(parents=True, exist_ok=True)

# Configuration dictionary
cf = {
    "image_size": 256,
    "patch_size": 16,
    "num_channels": 3,
    "flattened_patch_dim": 16 * 16 * 3,
}

# Declare global variables

filepath1 = None  # Global variable for original file path
filepath2 = None  # Global variable for predicted file path
async def process_nii(file: UploadFile, model):
    """Full pipeline: slices -> predict masks -> rebuild NIfTI, with output in current directory."""
    # global filepath1, filepath2
    # filepath1 = "original_3b645fbd29544b7096497bf629ec093b_IMG_0002.nii.gz"
    # filepath2 = "predicted_5a7728a1e4b346b1bba32c95b6f65fe4_IMG_0002.nii.gz"
    # return {
    #         "success": True,
    #         "mask_path": '/predicted_nii/predicted_5a7728a1e4b346b1bba32c95b6f65fe4_IMG_0002.nii.gz',
    #         "original_path": '/original_nii/original_3b645fbd29544b7096497bf629ec093b_IMG_0002.nii.gz'
    #     }
    input_filename = f"uploaded_{uuid.uuid4().hex}_{file.filename}"
    input_nii_path = UPLOAD_FOLDER / input_filename
    predicted_nii_filename = f"predicted_{uuid.uuid4().hex}_{file.filename}"
    predicted_nii_path_server = PREDICTED_NII_PUBLIC_FOLDER / predicted_nii_filename  # Changed
    original_filename_moved = f"original_{uuid.uuid4().hex}_{file.filename}"
    original_file_path_moved = ORIGINAL_NII_PUBLIC_FOLDER / original_filename_moved

    # Clean previous slices and masks for this processing
    shutil.rmtree(SLICES_FOLDER, ignore_errors=True)
    SLICES_FOLDER.mkdir(exist_ok=True)
    shutil.rmtree(OUTPUT_MASKS_FOLDER, ignore_errors=True)
    OUTPUT_MASKS_FOLDER.mkdir(exist_ok=True)

    try:
        # Save the uploaded file temporarily
        with open(input_nii_path, "wb") as f:
            while contents := await file.read(1024 * 1024):
                f.write(contents)

        nii_img = nib.load(input_nii_path)
        nii_data = nii_img.get_fdata()
        depth = nii_data.shape[0]  # slices along axis 0

        # Step 1: Slice the NIfTI
        for i in range(depth):
            slice_img = nii_data[i, :, :]
            slice_img = np.clip(slice_img, 0, 255).astype(np.uint8)
            slice_resized = cv2.resize(slice_img, (512, 512), interpolation=cv2.INTER_AREA)
            slice_rgb = cv2.cvtColor(slice_resized, cv2.COLOR_GRAY2BGR)
            slice_path = SLICES_FOLDER / f"slice_{i:03d}.png"
            cv2.imwrite(str(slice_path), slice_rgb)

        # Step 2: Predict masks
        image_files = sorted(os.listdir(SLICES_FOLDER))
        predicted_mask_paths = []

        for image_name in image_files:
            input_image_path = SLICES_FOLDER / image_name
            image = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            resized = cv2.resize(image, (cf["image_size"], cf["image_size"]), interpolation=cv2.INTER_LANCZOS4)
            norm = resized / 255.0
            patches = patchify(norm, (cf["patch_size"], cf["patch_size"], cf["num_channels"]), cf["patch_size"])
            patches = patches.reshape(-1, cf["flattened_patch_dim"])
            patches = np.expand_dims(patches, axis=0)
            pred = model.predict(patches, verbose=0)[0]
            pred = (pred * 255).astype(np.uint8)
            if len(pred.shape) == 3 and pred.shape[-1] > 1:
                pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
            mask_resized = cv2.resize(pred, (cf["image_size"], cf["image_size"]), interpolation=cv2.INTER_NEAREST)
            _, mask_thresh = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
            output_mask_path = OUTPUT_MASKS_FOLDER / f"mask_{image_name}"
            cv2.imwrite(str(output_mask_path), mask_thresh)
            predicted_mask_paths.append(output_mask_path)

        # Step 3: Stack masks back into a NIfTI
        mask_files = sorted(os.listdir(OUTPUT_MASKS_FOLDER), key=lambda x: int("".join(filter(str.isdigit, x))))
        mask_slices = []

        for fname in mask_files:
            path = OUTPUT_MASKS_FOLDER / fname
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
            mask_slices.append(resized)

        if not mask_slices:
            raise Exception("No masks were generated.")

        predicted_volume = np.stack(mask_slices, axis=0)
        nii_pred = nib.Nifti1Image(predicted_volume, affine=np.eye(4))
        nib.save(nii_pred, predicted_nii_path_server)  # Save to the new location
        predicted_file_url = f"/predicted_nii/{predicted_nii_filename}"  #changed
        # Step 4: Move the original uploaded file to the public folder
        # original_filename_moved = f"original_{uuid.uuid4().hex}_{file.filename}"
        # original_file_path_moved = ORIGINAL_NII_PUBLIC_FOLDER / original_filename_moved # changed
        shutil.move(str(input_nii_path), str(original_file_path_moved))  # Save to the new location.
        original_file_url = f"/{original_filename_moved}" 
        predicted_file_url = f"/{predicted_nii_filename}" # changed
        global filepath1, filepath2
        filepath1 = original_file_url
        filepath2 = predicted_file_url
        print(filepath1)
        print({"success": True, "mask_path": predicted_file_url, "original_path": original_file_url})
        return {"success": True, "mask_path": predicted_file_url, "original_path": original_file_url}

    except nib.NiftiError as e:
        print(f"NiBabel Error: {e}")
        return {"success": False, "error": f"NiBabel Error: {e}"}
    except Exception as e:
        print(f"Error processing NII file: {e}")
        return {"success": False, "error": f"Error processing NII file: {e}"}
    finally:
        # Clean up the temporary uploaded file (if not moved) and temp folders
        if input_nii_path.exists():
            os.remove(input_nii_path)
        for folder in [SLICES_FOLDER, OUTPUT_MASKS_FOLDER]:
            if folder.exists():
                for file_path in folder.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)

docking_cache = {}
docking_cache_file = "docking_cache.pkl"

def load_docking_cache():
    global docking_cache
    if Path(docking_cache_file).exists():
        docking_cache = joblib.load(docking_cache_file)

def save_docking_cache():
    joblib.dump(docking_cache, docking_cache_file)

load_docking_cache()

def run_pdb2pqr(pdb_id,ligand_id):
    # Define paths
    input_pdb = f"protein_structures_final/{pdb_id}.pdb"
    output_pqr = f"protein_structures_final/protein_{pdb_id}.pqr"
    output_pdb = f"protein_structures_final/protein_{pdb_id}_h.pdb"
    
    pdb2pqr_path = r"C:\Users\Srinidhi\AppData\Roaming\Python\Python39\Scripts\pdb2pqr.exe"

    # Check if the input PDB file exists
    if not os.path.exists(input_pdb):
        print(f"Error: Input PDB file '{input_pdb}' does not exist.")
        return

    # Check if the output folder exists, if not, create it
    output_folder = "protein_structures_final"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # Check if the pdb2pqr executable exists
    if not os.path.exists(pdb2pqr_path):
        print(f"Error: 'pdb2pqr.exe' not found at {pdb2pqr_path}.")
        return

    # Define the command to run
    command = [
        pdb2pqr_path,  # Full path to pdb2pqr
        f"--pdb-output={output_pdb}",
        "--pH=7.4",
        input_pdb,
        output_pqr,
        "--whitespace"
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"Successfully processed PDB ID: {pdb_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error running pdb2pqr for {pdb_id}: {e}")
    #     # Step 1: Compute pocket center and box using MDAnalysis
    u = mda.Universe(f"protein_structures_final/protein_{pdb_id}.pqr")
    u.atoms.write(f"pdbqt_final/{pdb_id}.pdbqt")
    with open(f"pdbqt_final/{pdb_id}.pdbqt", 'r') as f:
        lines = f.readlines()

    with open(f"pdbqt_final/{pdb_id}.pdbqt", 'w') as f:
        f.writelines(lines[2:])
    structure = mda.Universe(f"protein_structures_final/{pdb_id}.pdb")
    # Changed to upper case
    ligand_atoms = structure.select_atoms(f"resname {ligand_id.upper()}")

    pocket_center = ligand_atoms.center_of_geometry().tolist()
    ligand_box = (ligand_atoms.positions.max(axis=0) - ligand_atoms.positions.min(axis=0) + 5).tolist()
    os.makedirs("docking_results",exist_ok=True)
    # Step 2: Run AutoDock Vina
    receptor_path = f"pdbqt_final/{pdb_id}.pdbqt"
    ligand_path = f"pdbqt_final/{ligand_id}.pdbqt"
    output_path = f"docking_results/{ligand_id}_output.pdbqt"
    csv_output_path = f"docking_results/{pdb_id}_{ligand_id}_docking.csv"

    vina_command = [
        "vina/vina_1.2.7_win.exe",
        "--receptor", receptor_path,
        "--ligand", ligand_path,
        "--center_x", str(pocket_center[0]),
        "--center_y", str(pocket_center[1]),
        "--center_z", str(pocket_center[2]),
        "--size_x", str(ligand_box[0]),
        "--size_y", str(ligand_box[1]),
        "--size_z", str(ligand_box[2]),
        "--exhaustiveness", "5",
        "--num_modes", "5",
        "--out", output_path
    ]
    subprocess.run(vina_command, check=True)

    # Step 3: Parse docking results
    with open(output_path, 'r') as f:
        lines = f.readlines()

    results = []
    total = inter = intra = torsions = None
    intra_best_pose = None
    recording = False

    for line in lines:
        if line.startswith("REMARK VINA RESULT:"):
            if total is not None:
                torsions = total - (inter + intra)
                results.append([total, inter, intra, torsions, intra_best_pose])
            parts = line.strip().split()
            total = float(parts[3])
            inter = intra = None
            recording = True
        elif recording and line.startswith("REMARK INTER:"):
            inter = float(line.strip().split()[-1])
        elif recording and line.startswith("REMARK INTRA:"):
            intra = float(line.strip().split()[-1])
        elif intra_best_pose is None and line.startswith("REMARK UNBOUND:"):
            intra_best_pose = float(line.strip().split()[-1])

    if total is not None and inter is not None and intra is not None:
        torsions = total - (inter + intra)
        results.append([total, inter, intra, torsions, intra_best_pose])

    # Step 4: Save and return DataFrame
    df = pd.DataFrame(results, columns=["total", "inter", "intra", "torsions", "intra best pose"])
    df.to_csv(csv_output_path, index=False)
    print(f"Energies CSV file saved at {csv_output_path}")
    return df,csv_output_path   
# --- FastAPI Integration ---


app = FastAPI()
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins as per your requirement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

num_features = 4
num_actions = 4  
state_size = 4
class DockingData(BaseModel):
    ec_number: str
    ligand_id: str

class InputData(BaseModel):
    SMILES: str

class ProteinData(BaseModel):
    PROTEIN: str

@app.post("/predict/reinforcement")
def reinforcement_learning_generate(data: InputData):
    try:
        input_smiles = data.SMILES.strip()

        if not input_smiles:
            raise HTTPException(status_code=400, detail="SMILES input missing")

        # === Step 1: Check Cache ===
        if input_smiles in reinforcement_generation_cache:
            print("Returning cached result for:", input_smiles)
            cached_results = reinforcement_generation_cache[input_smiles]
            best = cached_results[0]
            return JSONResponse(content={
                'top_results': cached_results,
                'best_smile': best.get('SMILES'),
                'best_reward': best.get('Reward')
            })

        # === Step 2: No cache -> Generate fresh ===
        user_smile = '<' + input_smiles
        results = []
        seen = set()
        count = 0

        while count < 5:
            smile = my_generator.evaluate(gen_data, prime_str=user_smile)
            print('Generated molecules', count)
            try:
                mol = Chem.MolFromSmiles(smile[1:-1])
                valid_smile = smile[1:-1]
                if mol and valid_smile not in seen:
                    seen.add(valid_smile)
                    prediction_result = predictor_pic50.predict([valid_smile])
                    pi = prediction_result[1]

                    if isinstance(pi, torch.Tensor):
                        pi = pi.cpu().numpy()
                        reward = pi.item() if pi.size == 1 else pi.tolist()
                    elif isinstance(pi, list) and pi:
                        try:
                            reward = float(pi[0])
                        except Exception as e:
                            print("List prediction error:", e)
                            reward = None
                    else:
                        reward = None

                    results.append({"SMILES": valid_smile, "Reward": reward})
                    count += 1

            except Exception as e:
                print(f"Error processing SMILE: {smile} - {e}")
                continue

        sorted_results = sorted(results, key=lambda x: x['Reward'] or 0, reverse=True)
        best = sorted_results[0] if sorted_results else {}

        # === Step 3: Cache the result ===
        reinforcement_generation_cache[input_smiles] = sorted_results
        save_cache()

        # return JSONResponse(content={
        #     'top_results': sorted_results,
        #     'best_smile': best.get('SMILES'),
        #     'best_reward': best.get('Reward')
        # })
        return {
            "top_results": sorted_results
        }

    except Exception as e:
        print("Unexpected error:", e)
        raise HTTPException(status_code=500, detail=str(e))
# @app.post("/predict/reinforcement")
# def predict(data: InputData):
#     """
#     Endpoint to run the simulation using the input SMILES string.
#     The SMILES string (provided as y_train) is processed to compute its molecular properties.
#     The environment target is updated based on these properties.
#     """
#     user_smile = data.SMILES
#     user_smile=user_smile.strip()
#     user_smile='<'+user_smile
#     print(user_smile)
#     count =0
#     results=[]
#     s=set()

#     while count < 5:
#         smile = my_generator.evaluate(gen_data, prime_str=user_smile)
#         print('Generated molecules ', count)
#         try:
#             mol = Chem.MolFromSmiles(smile[1:-1])
#             valid_smile = smile[1:-1]
#             if mol and valid_smile not in s:
#                 s.add(valid_smile)
#                 prediction_result = predictor_pic50.predict([valid_smile])
#                 pi = prediction_result[1]  # Get the prediction value

#                 if isinstance(pi, torch.Tensor):
#                     pi_numpy_array = pi.cpu().numpy()
#                     pi_float = pi_numpy_array.item() if pi_numpy_array.size == 1 else pi_numpy_array.tolist()
#                     results.append({"SMILES": valid_smile, "Reward": pi_float})
#                     print(f"Tensor Prediction (Float): {pi_float}")
#                 elif isinstance(pi, list):
#                     if pi:
#                         try:
#                             pi_float = float(pi[0])
#                             results.append({"SMILES": valid_smile, "Reward": pi_float})
#                             print(f"List Prediction (Float): {pi_float}")
#                         except (ValueError, TypeError, IndexError) as e:
#                             print(f"Error processing list prediction: {pi} - {e}")
#                             results.append({"SMILES": valid_smile, "Reward": None})
#                     else:
#                         print("Empty list prediction.")
#                         results.append({"SMILES": valid_smile, "Reward": None})
#                 else:
#                     print(f"Unexpected prediction type: {type(pi)}, value: {pi}")
#                     results.append({"SMILES": valid_smile, "Reward": None})

#                 count += 1

#         except Exception as e:
#             print(f"Error validating SMILES: {smile} - {e}")


#     print(results)

#     rs = sorted(results, key=lambda item: item.get("Reward"), reverse=True)
    
#     print(rs)
#     # Return the simulation output along with the processed top 5 results.
#     return {
#         "top_results": rs
#     }

@app.post("/predict/reinforcement/logp")
def predict(data: InputData):
    """
    Endpoint to run the simulation using the input SMILES string.
    The SMILES string (provided as y_train) is processed to compute its molecular properties.
    The environment target is updated based on these properties.
    """
    user_smile = data.SMILES
    user_smile=user_smile.strip()
    user_smile='<'+user_smile
    print(user_smile)
    count =0
    results=[]
    s=set()

    while count < 5:
        smile = my_generator_logp.evaluate(gen_data, prime_str=user_smile)
        print('Generated molecules ', count)
        try:
            mol = Chem.MolFromSmiles(smile[1:-1])
            valid_smile = smile[1:-1]
            if mol and valid_smile not in s:
                s.add(valid_smile)
                prediction_result = predictor_logP.predict([valid_smile])
                pi = prediction_result[1]  # Get the prediction value

                if isinstance(pi, torch.Tensor):
                    pi_numpy_array = pi.cpu().numpy()
                    pi_float = pi_numpy_array.item() if pi_numpy_array.size == 1 else pi_numpy_array.tolist()
                    results.append({"SMILES": valid_smile, "Reward": pi_float})
                    print(f"Tensor Prediction (Float): {pi_float}")
                elif isinstance(pi, list):
                    if pi:
                        try:
                            pi_float = float(pi[0])
                            results.append({"SMILES": valid_smile, "Reward": pi_float})
                            print(f"List Prediction (Float): {pi_float}")
                        except (ValueError, TypeError, IndexError) as e:
                            print(f"Error processing list prediction: {pi} - {e}")
                            results.append({"SMILES": valid_smile, "Reward": None})
                    else:
                        print("Empty list prediction.")
                        results.append({"SMILES": valid_smile, "Reward": None})
                else:
                    print(f"Unexpected prediction type: {type(pi)}, value: {pi}")
                    results.append({"SMILES": valid_smile, "Reward": None})

                count += 1

        except Exception as e:
            print(f"Error validating SMILES: {smile} - {e}")


    print(results)

    rs = sorted(results, key=lambda item: item.get("Reward"), reverse=True)
    
    print(rs)
    # Return the simulation output along with the processed top 5 results.
    return {
        "top_results": rs
    }
@app.post("/predict/segmentation")
async def predict_segmentation(
    image_file: UploadFile = File(...),
    model_type: str = Form(...)
):
    try:
        print(f"Received model_type: {model_type}")
        print(f"Received image_file: {image_file.filename}, {image_file.content_type}")

        model, error = load_model(model_type)
        if error:
            raise HTTPException(status_code=500, detail=error)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found.")

        mask_path, error_process = await process_image(image_file, model)
        if error_process  :
            return {"success": True, "mask_path": mask_path}
        
        if error_process!=True:
            raise HTTPException(status_code=500, detail=error_process)
        if not mask_path:
            raise HTTPException(status_code=500, detail="Failed to process and save the prediction mask.")

        

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
@app.post("/predict/uploadnii")
async def predict_segmentation(
    image_file: UploadFile = File(...),
    
):
    try:
        print(f"Received model_type: Lung")
        print(f"Received image_file: {image_file.filename}, {image_file.content_type}")

        model, error = load_model("lung")
        if error:
            raise HTTPException(status_code=500, detail=error)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found.")

        response = await process_nii(image_file, model)
        if response["success"]:
            return {"success": True, "mask_path": response["mask_path"], "original_path": response["original_path"]}

        
        if response["success"]!=True:
            raise HTTPException(status_code=500, detail=error_process)
        if not response["mask_path"]:
            raise HTTPException(status_code=500, detail="Failed to process and save the prediction mask.")

        

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
@app.post("/predict/protein2smiles")
def predict(data: ProteinData):
    try:
        print(data)
        protein_seq = data.PROTEIN
        protein_seq=protein_seq.strip()
        print(protein_seq)
        

        if not protein_seq:
            raise HTTPException(status_code=400, detail="No protein sequence provided,")
        print(protein_seq)
        smiles_str = predict_smiles(protein_seq)
        result={
            "SMILES": smiles_str,
            
        }
        return {
        "top_results": result
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    
@app.post("/predict/mask")
def predict(data: InputData):
    user_smile = data.SMILES
    user_smile=user_smile.strip()
    predicted_smiles = chemberta_predict(user_smile,tokenizer, vocab,device,chembert_model)
    print(predicted_smiles)
    return {
        
        "top_results": predicted_smiles
        
    }
@app.post('/predict/vit')
def predict(data : InputData):
    try:
        print("hi10")
        smiles = data.SMILES
        
        # Convert SMILES to fingerprint
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES")
        print(mol,smiles)
        print("hi11")  
        try:
            # print("hero")
            # fpgen = GetMorganGenerator(radius=2, fpSize=1024)
            # print("hi112")  
            # fingerprint = fpgen.GetFingerprint(mol)
            # print("hi1")
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            print("1")
        except Exception as  e:
            raise HTTPException(status_code=400, detail="error during genrating fps")
        fp_array = np.array(fingerprint, dtype=np.float32)
        fp_array = fp_array / (fp_array.max() + 1e-6)
        print("hi2")
        # Convert to image and predict
        image = morgan_to_image(fp_array)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        print("hi3")
        with torch.no_grad():
            output = vit(image_tensor)
            predicted_class_idx = output.argmax(1).item()
            predicted_label = le.inverse_transform([predicted_class_idx])[0]
        print("hi4")
        result={
            "smiles": smiles,
            "predicted_class": predicted_label
        }
        print(result)
        return result
            
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/predict/docking")
async def visualize_interaction(data: DockingData):
    logger.info(f"Starting visualize_interaction with data: {data}")
    try:
        # Step 1: Check Cache
        cache_key = f"{data.ec_number}_{data.ligand_id}"
        if cache_key in docking_cache:
            logger.info(f"Returning cached docking result for: {cache_key}")
            cached_result = docking_cache[cache_key]
            return {
                "visualization_html": cached_result["visualization_html"],
                "energy_table": cached_result["energy_table"]
            }

        protein_dir = os.path.join(os.getcwd(), "protein_structures_final")
        ligand_dir = os.path.join(os.getcwd(), "ligands_to_dock_final")
        pdbqt_dir = os.path.join(os.getcwd(), "pdbqt_final")
        os.makedirs(protein_dir, exist_ok=True)
        os.makedirs(ligand_dir, exist_ok=True)
        os.makedirs(pdbqt_dir, exist_ok=True)

        logger.info(f"Downloading PDB and ligand for EC: {data.ec_number}, Ligand: {data.ligand_id}")
        pdb_id, ligand_id_lower = download_pdb_and_ligand(data.ec_number, data.ligand_id, protein_dir, ligand_dir)
        if not pdb_id or not ligand_id_lower:
            logger.error("Failed to download PDB or Ligand files.")
            raise HTTPException(status_code=404, detail="Failed to download PDB or Ligand files.")

        logger.info(f"Converting ligand {ligand_id_lower} to pdbqt")
        pdbqt_errors = convert_ligand_to_pdbqt(ligand_id_lower,pdb_id ,ligand_dir,protein_dir, pdbqt_dir)
        logger.info(f"Creating Energies CSV table ... ")
        df, csv_file_path =run_pdb2pqr(pdb_id, ligand_id_lower)
        df_json = df.to_dict(orient="records")
        logger.info(f"Adding hydrogens to protein {pdb_id}")
        add_hydrogens_to_protein(pdb_id)

        logger.info(f"Running PLIF and visualization for PDB: {pdb_id}, Ligand: {ligand_id_lower}")
        html_content, plif_errors = await run_plif_and_visualize(pdb_id, ligand_id_lower, protein_dir, ligand_dir)

        if html_content:
            docking_cache[cache_key] = {
        "visualization_html": html_content,
        "energy_table": df_json
    }
            save_docking_cache()
            logger.info("Successfully generated ProLIF visualization.")
            return {"visualization_html": html_content,
                    "energy_table": df_json}
        else:
            logger.error(f"Failed to generate ProLIF visualization. Errors: {plif_errors}")
            raise HTTPException(status_code=500, detail="Failed to generate ProLIF visualization.")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
app.mount("/original_nii", StaticFiles(directory="static/original_nii"), name="original_nii")
app.mount("/predicted_nii", StaticFiles(directory="static/predicted_nii"), name="predicted_nii")
templates = Jinja2Templates(directory="templates")
@app.get('/papaya', response_class=HTMLResponse)
async def papaya_viewer(request: Request):
    """
    Displays the Papaya viewer page, passing the global file paths.
    """
    global filepath1, filepath2  # Declare globals
    print(filepath1 )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "file1_path": filepath1,
            "file2_path": filepath2,
        },
    )
# For running with: uvicorn filename:app --reload
print(app.routes)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)