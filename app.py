import streamlit as st
import pandas as pd
import ast
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import streamlit.components.v1 as components
import py3Dmol

# --- 1. App Configuration ---
st.set_page_config(page_title="Crystal 3D Surface Explorer", layout="wide")
st.title("🧬 Acid-Base 3D Surface Mapper")

# --- 2. Load Database (Cached for Speed) ---
@st.cache_data
def load_database():
    # Update the URL to point to the new .gz file
    url = "https://raw.githubusercontent.com/BencenoMrX/pKaLGBM/main/joint_database.csv.gz"
    try:
        # Pandas handles the decompression automatically
        df = pd.read_csv(url, compression='gzip')
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return pd.DataFrame()
   

# --- 3. Cheminformatics Engine ---
def inject_experimental_coords(mol, coords_string):
    """Parses raw [(x,y,z)] strings and injects them into an RDKit Mol."""
    try:
        # Safely convert the string "[(1,2,3), ...]" into a real Python list of tuples
        coords = ast.literal_eval(coords_string)
        
        # Ensure the number of heavy atoms in the SMILES matches the coordinates
        heavy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        if len(heavy_atoms) != len(coords):
            return None
            
        # Create an empty 3D Conformer
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.Set3D(True)
        
        # Map the x, y, z coordinates to the RDKit atom indices
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, Point3D(x, y, z))
            
        # Attach the 3D coordinates to the molecule
        mol.AddConformer(conf, assignId=True)
        return mol
    except:
        return None

def build_3d_molecule(smiles, df, use_experimental=True):
    """Builds the 3D PDB block, handling missing hydrogens and coordinate logic."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    success_experimental = False
    
    # Try to find and apply experimental coordinates
    if use_experimental and not df.empty:
        # Find the row where the SMILES matches either column 1 or 2
        match = df[(df['smiles1'] == smiles) | (df['smiles2'] == smiles)]
        
        if not match.empty:
            row = match.iloc[0]
            
            # Intelligently grab the correct coordinate array
            if row['smiles1'] == smiles:
                coords_string = row['coordinates1']
            else:
                coords_string = row['coordinates2']
                
            # Make sure coordinates actually exist (aren't NaN)
            if pd.notna(coords_string):
                mol_with_coords = inject_experimental_coords(mol, coords_string)
                if mol_with_coords:
                    mol = mol_with_coords
                    # Add hydrogens based on the heavy atom framework
                    mol = Chem.AddHs(mol, addCoords=True)
                    success_experimental = True
                    st.success(f"Loaded Experimental Coordinates for: {smiles}")

    # Fallback to RDKit Generation if requested or if experimental failed
    if not success_experimental:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        if use_experimental:
            st.warning(f"No valid experimental coords found. Generated 3D for: {smiles}")
        else:
            st.info(f"Generated RDKit 3D conformation for: {smiles}")
            
    # Calculate Charges and inject into the B-factor (temperature) column
    AllChem.ComputeGasteigerCharges(mol)
    for atom in mol.GetAtoms():
        charge = atom.GetProp('_GasteigerCharge')
        charge = float(charge) if str(charge) != 'nan' else 0.0
            
        pdb_info = Chem.AtomPDBResidueInfo()
        pdb_info.SetTempFactor(charge)
        atom.SetPDBResidueInfo(pdb_info)
        
    return Chem.MolToPDBBlock(mol)

def render_3d_surface(pdb_block1, pdb_block2=None):
    """Renders the 3D py3Dmol viewer."""
    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_block1, 'pdb')
    if pdb_block2:
        view.addModel(pdb_block2, 'pdb')
        
    view.setStyle({'stick': {'radius': 0.15}})
    view.addSurface(py3Dmol.VDW, 
                    {'opacity': 0.8, 
                     'colorscheme': {'prop': 'b', 'gradient': 'rwb', 'min': -0.3, 'max': 0.3}})
    view.zoomTo()
    return view

# --- 4. Streamlit UI Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input Molecules")
    mode = st.radio("Select Mode:", ["Single Molecule", "Molecule Pair (Cocrystal)"])
    
    # The new toggle switch!
    use_experimental = st.toggle("Use Experimental Coordinates (if available)", value=True)
    
    smiles1 = st.text_input("SMILES 1", value="C1=CC=C(C=C1)C(=O)O") 
    smiles2 = ""
    
    if mode == "Molecule Pair (Cocrystal)":
        smiles2 = st.text_input("SMILES 2", value="C1=CC=NC=C1")
        
    generate_btn = st.button("Generate 3D Surface", type="primary")

with col2:
    if generate_btn:
        with st.spinner("Processing 3D structures and electrostatic mapping..."):
            pdb1 = build_3d_molecule(smiles1, df_joint, use_experimental)
            
            if mode == "Molecule Pair (Cocrystal)" and smiles2:
                pdb2 = build_3d_molecule(smiles2, df_joint, use_experimental)
                if pdb1 and pdb2:
                    view = render_3d_surface(pdb1, pdb2)
                    components.html(view._make_html(), height=500, width=800)
            else:
                if pdb1:
                    view = render_3d_surface(pdb1)
                    components.html(view._make_html(), height=500, width=800)
