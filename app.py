import streamlit as st
import pandas as pd
import ast
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import streamlit.components.v1 as components
import py3Dmol
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- 1. App Configuration ---
st.set_page_config(page_title="Crystal 3D Surface Explorer", layout="wide")
st.title("🧬 Acid-Base 3D Surface Mapper")

# --- 2. Load Database (Cached for Speed) ---
@st.cache_data
def load_database():
    url = "https://raw.githubusercontent.com/BencenoMrX/pKaLGBM/main/joint_database.csv.gz"
    try:
        df = pd.read_csv(url, compression='gzip')
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return pd.DataFrame()

df_joint = load_database()

# --- 3. Cheminformatics Engine ---
def inject_experimental_coords(mol, coords_string):
    """Parses raw [(x,y,z)] strings and injects them into an RDKit Mol."""
    try:
        coords = ast.literal_eval(coords_string)
        heavy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        if len(heavy_atoms) != len(coords):
            return None
            
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.Set3D(True)
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, Point3D(x, y, z))
            
        mol.AddConformer(conf, assignId=True)
        return mol
    except:
        return None

def build_3d_molecule(smiles, df, use_experimental=True):
    """Builds a MolBlock and extracts Gasteiger charges as a list."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None, None
    
    success_experimental = False
    
    if use_experimental and not df.empty:
        match = df[(df['smiles1'] == smiles) | (df['smiles2'] == smiles)]
        if not match.empty:
            row = match.iloc[0]
            if row['smiles1'] == smiles:
                coords_string = row['coordinates1']
            else:
                coords_string = row['coordinates2']
                
            if pd.notna(coords_string):
                mol_with_coords = inject_experimental_coords(mol, coords_string)
                if mol_with_coords:
                    mol = mol_with_coords
                    mol = Chem.AddHs(mol, addCoords=True)
                    success_experimental = True
                    st.success(f"Loaded Experimental Coordinates for: {smiles}")

    if not success_experimental:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        if use_experimental:
            st.warning(f"No valid experimental coords found. Generated 3D for: {smiles}")
        else:
            st.info(f"Generated RDKit 3D conformation for: {smiles}")
            
    # Calculate Charges
    AllChem.ComputeGasteigerCharges(mol)
    charges = []
    for atom in mol.GetAtoms():
        charge = atom.GetProp('_GasteigerCharge')
        charges.append(float(charge) if str(charge) != 'nan' else 0.0)
        
    # Return standard MolBlock instead of PDB
    return Chem.MolToMolBlock(mol), charges

def render_3d_surface(mb1, charges1, mb2=None, charges2=None):
    """Renders the 3D py3Dmol viewer using Matplotlib colormaps."""
    view = py3Dmol.view(width=800, height=500)
    
    # We use bwr_r (Blue-White-Red reversed) so Negative (-0.3) is Red, Positive (+0.3) is Blue
    cmap = cm.get_cmap('bwr_r')
    norm = mcolors.Normalize(vmin=-0.3, vmax=0.3)
    
    # Render Molecule 1
    view.addModel(mb1, 'sdf')
    color_map1 = {i: mcolors.to_hex(cmap(norm(c))) for i, c in enumerate(charges1)}
    
    view.setStyle({'model': 0}, {'stick': {'radius': 0.15}})
    view.addSurface(py3Dmol.VDW, 
                    {'opacity': 0.8, 'colorscheme': {'prop': 'index', 'map': color_map1}}, 
                    {'model': 0})
    
    # Render Molecule 2 if it exists
    if mb2 and charges2:
        view.addModel(mb2, 'sdf')
        color_map2 = {i: mcolors.to_hex(cmap(norm(c))) for i, c in enumerate(charges2)}
        
        view.setStyle({'model': 1}, {'stick': {'radius': 0.15}})
        view.addSurface(py3Dmol.VDW, 
                        {'opacity': 0.8, 'colorscheme': {'prop': 'index', 'map': color_map2}}, 
                        {'model': 1})
        
    view.zoomTo()
    return view

# --- 4. Streamlit UI Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input Molecules")
    mode = st.radio("Select Mode:", ["Single Molecule", "Molecule Pair (Cocrystal)"])
    use_experimental = st.toggle("Use Experimental Coordinates (if available)", value=True)
    
    smiles1 = st.text_input("SMILES 1", value="C1=CC=C(C=C1)C(=O)O") 
    smiles2 = ""
    if mode == "Molecule Pair (Cocrystal)":
        smiles2 = st.text_input("SMILES 2", value="C1=CC=NC=C1")
        
    generate_btn = st.button("Generate 3D Surface", type="primary")

with col2:
    if generate_btn:
        with st.spinner("Processing 3D structures and electrostatic mapping..."):
            mb1, charges1 = build_3d_molecule(smiles1, df_joint, use_experimental)
            
            if mode == "Molecule Pair (Cocrystal)" and smiles2:
                mb2, charges2 = build_3d_molecule(smiles2, df_joint, use_experimental)
                if mb1 and mb2:
                    view = render_3d_surface(mb1, charges1, mb2, charges2)
                    components.html(view._make_html(), height=500, width=800, scrolling=False)
            else:
                if mb1:
                    view = render_3d_surface(mb1, charges1)
                    components.html(view._make_html(), height=500, width=800, scrolling=False)
