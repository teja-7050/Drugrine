import os
import requests
import MDAnalysis as mda
import subprocess
import prolif as plf
from rdkit import Chem
from rcsbsearchapi import rcsb_attributes as attrs
from rcsbsearchapi.search import TextQuery
import warnings 
import traceback
import asyncio
import pandas as pd
from MDAnalysis.topology import tables  # This will trigger the DeprecationWarning
from prolif.interactions import HBDonor # This will trigger a UserWarning

warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.topology.tables")
warnings.filterwarnings("ignore", message="The 'HBDonor' interaction has been superseded", category=UserWarning, module="prolif.interactions.base")
# Add similar filters for other prolif UserWarnings if needed

# Your code that uses MDAnalysis and prolif here

def download_pdb_and_ligand(ECnumber, LIGAND_ID,protein_dir="protein_structures_final", ligand_dir="ligands_to_dock_final"):
    q1 = attrs.rcsb_polymer_entity.rcsb_ec_lineage.id == ECnumber  
    q2 = TextQuery(LIGAND_ID)

    query = q1 & q2             

    results = list(query())
    if not results:
        print("No results found.")
        return None, None
    
    pdb_id = results[0].lower()  # Get the PDB ID and convert to lowercase
    print(f"PDB ID: {pdb_id}")
    
    ligand_id = LIGAND_ID.lower()   
    print(f"Ligand ID: {ligand_id}")

    
    os.makedirs(protein_dir, exist_ok=True)

    pdb_request = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
    ligand_request = requests.get(f"https://files.rcsb.org/ligands/download/{ligand_id}_ideal.sdf")
    
    print("Protein:", pdb_request.status_code)
    print("Ligand:", ligand_request.status_code)

    with open(f"{protein_dir}/{pdb_id}.pdb", "w+") as f:
        f.write(pdb_request.text)
    
    os.makedirs(ligand_dir, exist_ok=True)
    with open(f"{ligand_dir}/{ligand_id}_ideal.sdf", "w+") as file:
        file.write(ligand_request.text)

    return pdb_id, ligand_id


def add_hydrogens_to_protein(pdb_id):
    output_pdb_h = f"protein_structures_final/protein_{pdb_id}_h.pdb"
    input_pdb = f"protein_structures_final/protein_{pdb_id}.pdb"
    
    # Ensure reduce tool is installed and available in the system
    subprocess.run([
        "reduce", 
        "-H",  # Adds hydrogens
        input_pdb,
        ">",  # Redirects output to a file
        output_pdb_h
    ], shell=True)
    print(f"Hydrogens added to protein {pdb_id}, saved to {output_pdb_h}")

import subprocess
import os

def convert_ligand_to_pdbqt(ligand_id, pdb_id, ligand_dir="ligands_to_dock_final", protein_dir="protein_structures_final", pdbqt_dir="pdbqt_final"):
    os.makedirs(pdbqt_dir, exist_ok=True)  # Ensure output dir exists

    # Convert ligand (.sdf) to .pdbqt
    subprocess.run([
        'C:/Program Files/OpenBabel-3.1.1/obabel.exe',
        f"{ligand_dir}/{ligand_id}_ideal.sdf",
        "-O", f"{pdbqt_dir}/{ligand_id}.pdbqt",
        "-p", "7.4", "--AddHydrogens"
    ], check=True)

    # Convert protein (.pdb) to .pdbqt with hydrogens
    # subprocess.run([
    #     "C:/Program Files/OpenBabel-3.1.1/obabel.exe",
    #     f"{protein_dir}/{pdb_id}.pdb",
    #     "-O", f"{pdbqt_dir}/{pdb_id}.pdbqt"

    # ], check=True)
    subprocess.run([
    "C:/Program Files/OpenBabel-3.1.1/obabel.exe",
    f"{protein_dir}/{pdb_id}.pdb",
    "-O", f"{pdbqt_dir}/{pdb_id}.pdbqt",
    "-p", "7.4", "--AddHydrogens", "--partialcharge", "gasteiger"
], check=True)


    print(f"Ligand {ligand_id} and protein {pdb_id} converted to PDBQT format.")
    return True

# async def visualize_interaction(pdb_id, ligand_id_lower, protein_dir, ligand_dir):
#     html_content, plif_errors = await run_plif_and_visualize(pdb_id, ligand_id_lower, protein_dir, ligand_dir)
#     # The rest of your visualize_interaction function remains the same
#     print(f"HTML Content: {html_content}")
#     if plif_errors:
#         print(f"PLIF Errors: {plif_errors}")
#     return html_content, plif_errors
##---------------------------------------------------------------------------------------------------------------

async def run_docking_and_extract(pdb_id, ligand_id):
    # Step 1: Compute pocket center and box using MDAnalysis
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
    return df, csv_output_path  # Return both the dataframe and the path



##---------------------------------------------------------------------------------------------------------------
async def run_plif_and_visualize(pdb_id, ligand_id, protein_dir="protein_structures_final", ligand_dir="ligands_to_dock_final"):
    print(protein_dir)
    pdb_file =  os.path.join(protein_dir, f"{pdb_id}.pdb")
               
    sdf_file = f"{ligand_dir}/{ligand_id}.sdf"
    # try:
    #     print("hi1")
    #     protein = mda.Universe(pdb_file)
    #     print("hi2")
    #     protein_plf = plf.Molecule.from_mda(protein, NoImplicit=False,)
    #     print("hi3")
    # except Exception as e:
    #     print("Error during MDAnalysis or ProLIF conversion:", e)
    #     traceback.print_exc()
    protein = mda.Universe(pdb_file)
    protein_plf = plf.Molecule.from_mda(protein, NoImplicit=False,)
    print("hi3")
    print("Protein loaded and converted to ProLIF format successfully.")
    input_file = f"ligands_to_dock_final/{ligand_id}_ideal.sdf"
    ligand_H = Chem.MolFromMolFile(input_file, removeHs=False)
    output_file = f"ligands_to_dock_final/{ligand_id}.sdf"
    Chem.MolToMolFile(ligand_H, output_file)

    poses_plf = list(plf.sdf_supplier(sdf_file))


    num_poses = len(poses_plf)
    print(f"Total poses available: {num_poses}")

    pose_index = 0
    if pose_index >= num_poses:
        raise IndexError(f"Invalid pose index: {pose_index}. Only {num_poses} poses available.")
    
    ligand_mol = poses_plf[pose_index]

    fp = plf.Fingerprint(count=True)
    print("Fingerprint Done --------------------")
    
    # Run fingerprint calculation
    fp.run_from_iterable(poses_plf, protein_plf,n_jobs=1)
    print(dir(fp)) 
    # ProLIF stores the results internally, so we need to use `fp.results`
    results = fp.ifp

    if not results:
        print("PLIF returned no results.")
        return None, "PLIF returned no results."
    else:
        print("PLIF calculation completed successfully.")
        
        # Get the first result (if available)
          # Or use any other logic to select the result
        
        # Visualize the interactions
        try:
            print(f"Ligand Molecule: {ligand_mol}")
            print(f"Protein Molecule: {protein_plf}")   
            view = fp.plot_3d(
            ligand_mol, protein_plf, frame=pose_index, display_all=False
            )
        # Save to HTML
            html_content = view._make_html()

    
            print("3D visualization genrated")
            return html_content, []
        except Exception as e:
            print('failed to genrate html content due to ',e)
            return None,e
