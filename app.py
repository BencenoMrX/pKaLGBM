import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import streamlit.components.v1 as components
import py3Dmol

# --- 1. Import Your Custom Predictor Class ---
# This looks for the pka_predictor.py file inside your GitHub repository
from pka_predictor import PkaPredictor

# --- 2. App Configuration & Layout ---
st.set_page_config(page_title="Crystal 3D Surface Explorer", layout="wide")
st.title("🧬 Multi-Component Crystal Engineering Suite")
st.markdown("Instantly predict $pK_a$ values and visualize 3D electrostatic surface maps.")

# --- 3. Initialize the ML Model (Cached) ---
@st.cache_resource
def init_predictor():
    try:
        # Assumes pka_lightgbm_model.pkl is in the same repo directory
        return PkaPredictor(model_path='pka_lightgbm_model.pkl')
    except Exception as e:
        st.error(f"Error loading pKa model weights: {e}")
        return None

predictor = init_predictor()

# --- 4. Color Grading Engine ---
def charge_to_hex(charge):
    """Maps charges from -0.3 (Deep Red, Basic) to +0.3 (Deep Blue, Acidic)."""
    c = max(-0.3, min(0.3, charge))
    val = c / 0.3  
    
    if val < 0:
        factor = 1.0 + val  
        r, g, b = 255, int(255 * factor), int(255 * factor)
    else:
        factor = 1.0 - val  
        r, g, b = int(255 * factor), int(255 * factor), 255
        
    return f"#{r:02x}{g:02x}{b:02x}"

def generate_html_view(smiles):
    """Generates a standard 3D MolBlock and maps color attributes by atom index."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
        
    # Generate clean, non-overlapping 3D coordinates using RDKit
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Calculate electronic properties
    AllChem.ComputeGasteigerCharges(mol)
    color_map = {}
    for i, atom in enumerate(mol.GetAtoms()):
        charge = atom.GetProp('_GasteigerCharge')
        charge_val = float(charge) if str(charge) != 'nan' else 0.0
        color_map[i] = charge_to_hex(charge_val)
        
    mb = Chem.MolToMolBlock(mol)
    
    # Construct the isolated 3D View
    view = py3Dmol.view(width=550, height=450)
    view.addModel(mb, 'sdf')
    view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'radius': 0.25}})
    view.addSurface(py3Dmol.VDW, {
        'opacity': 0.85, 
        'colorscheme': {'prop': 'index', 'map': color_map}
    })
    view.zoomTo()
    
    # Wrap with the vital 3Dmol Javascript engine script for desktop browser display
    html_code = f"""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
    <div style="width: 100%; height: 450px;">
        {view._make_html()}
    </div>
    """
    return html_code

# --- 5. Custom Color Bar Component ---
def display_color_scale():
    """Generates a native CSS colorbar legend mimicking the charge spectrum."""
    st.markdown("### 🎨 Charge Distribution Scale")
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; width: 100%; max-width: 500px; margin-bottom: 20px;">
            <div style="height: 25px; background: linear-gradient(to right, #ff0000, #ffffff, #0000ff); border-radius: 4px; border: 1px solid #ccc;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-top: 5px; font-weight: 500;">
                <span style="color: #d32f2f;">🔴 Basic (-0.3) <br><small>Proton Acceptor / Lone Pairs</small></span>
                <span style="color: #777;">Neutral (0.0)</span>
                <span style="color: #1976d2; text-align: right;">🔵 Acidic (+0.3) <br><small>Proton Donor / Acidic H</small></span>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- 6. Interface Assembly ---
# Left Sidebar for controls, Right panel for dynamic visualization windows
sidebar, display_panel = st.columns([1, 3])

with sidebar:
    st.subheader("Configuration Options")
    mode = st.radio("System Framework:", ["Single Molecule", "Cocrystal Pair"])
    
    smiles_input1 = st.text_input("SMILES Structure 1", value="C1=CC=C(C=C1)C(=O)O")
    smiles_input2 = ""
    if mode == "Cocrystal Pair":
        smiles_input2 = st.text_input("SMILES Structure 2", value="C1=CC=NC=C1")
        
    run_btn = st.button("Process System", type="primary")

with display_panel:
    if run_btn:
        # Validate entry 1
        m1_test = Chem.MolFromSmiles(smiles_input1)
        if not m1_test:
            st.error(f"❌ Structural parser failed. Please verify SMILES 1 string format.")
        
        # Validate entry 2 if in pair mode
        m2_test = None
        if mode == "Cocrystal Pair":
            m2_test = Chem.MolFromSmiles(smiles_input2)
            if not m2_test:
                st.error(f"❌ Structural parser failed. Please verify SMILES 2 string format.")

        # Process when inputs are validated
        if m1_test and (mode == "Single Molecule" or m2_test):
            display_color_scale()
            
            # --- Scenario A: Single Molecule Execution ---
            if mode == "Single Molecule":
                st.subheader("System Mapping Output")
                
                # Predict value using LightGBM pipeline
                pka_val = predictor.predict(smiles_input1) if predictor else "N/A"
                st.metric(label="Predicted Target $pK_a$", value=pka_val)
                
                # Render standalone window
                html_out = generate_html_view(smiles_input1)
                if html_out:
                    components.html(html_code=html_out, height=460, scrolling=False)
            
            # --- Scenario B: Pair Execution (Side-by-Side Windows) ---
            else:
                st.subheader("System Mapping Output")
                
                # Compute predictions for both structures
                pka_1 = predictor.predict(smiles_input1) if predictor else "N/A"
                pka_2 = predictor.predict(smiles_input2) if predictor else "N/A"
                
                # Establish dynamic window partition columns
                win_col1, win_col2 = st.columns(2)
                
                with win_col1:
                    st.markdown("#### Component A Structure")
                    st.metric(label="Component A Predicted $pK_a$", value=pka_1)
                    html_out1 = generate_html_view(smiles_input1)
                    if html_out1:
                        components.html(html_code=html_out1, height=460, scrolling=False)
                        
                with win_col2:
                    st.markdown("#### Component B Structure")
                    st.metric(label="Component B Predicted $pK_a$", value=pka_2)
                    html_out2 = generate_html_view(smiles_input2)
                    if html_out2:
                        components.html(html_code=html_out2, height=460, scrolling=False)
