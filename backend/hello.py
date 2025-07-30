#STEP-1: install pdb2pqr using pip install pdb2pqr
#STEP-2: find the path where it is saving and set the
#  path here like : pdb2pqr_path = r"C:\Users\Srinidhi\AppData\Roaming\Python\Python39\Scripts\pdb2pqr.exe"
#STEP-3:install vina from  here : https://github.com/ccsb-scripps/AutoDock-Vina/releases  install this ->vina_1.2.7_win.exe
#  and save it in a folder and specify the path here vina_command = [
#    here--->    "vina/vina_1.2.7_win.exe",
#       "--receptor", receptor_path,
#       "--ligand", ligand_path,
#STEP-4: add the function run_pdb2pqr in main.py/app.py    
#STEP-5: add this complete code  which is given down @app.post("/predict/docking") or u can simply add 
#    logger.info(f"Creating Energies CSV table ... ")
#   df, csv_file_path =run_pdb2pqr(pdb_id, ligand_id_lower) this snippet in ur existing @app.post("/predict/docking") and modfiy the return statement  f html_content:
#            logger.info("Successfully generated ProLIF visualization.")
#            return {"visualization_html": html_content,
#                    "energy_table": df_json}













import MDAnalysis as mda



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



#################################################################################################################################
#################################################################################################################################
#################################################################################################################################


@app.post("/predict/docking")
async def visualize_interaction(data: DockingData):
    logger.info(f"Starting visualize_interaction with data: {data}")
    try:
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