# Save this file as pka_predictor.py in your GitHub repo
import os
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

class PkaPredictor:
    def __init__(self, model_path='pka_lightgbm_model.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)
        
    def _extract_features(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            
            # 1. 1024-bit Morgan Fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fp_arr = np.zeros((1,), dtype=np.float32)
            Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
            
            # 2. Gasteiger Charges
            mol_h = Chem.AddHs(mol)
            AllChem.ComputeGasteigerCharges(mol_h)
            charges = [float(a.GetProp('_GasteigerCharge')) for a in mol_h.GetAtoms() 
                       if a.HasProp('_GasteigerCharge') and not np.isnan(float(a.GetProp('_GasteigerCharge')))]
            max_charge = max(charges) if charges else 0.0
            min_charge = min(charges) if charges else 0.0
            
            # 3. All RDKit Descriptors
            rdkit_desc_values = []
            for desc_name, desc_func in Descriptors._descList:
                try:
                    val = desc_func(mol)
                    rdkit_desc_values.append(0.0 if np.isnan(val) or np.isinf(val) else float(val))
                except:
                    rdkit_desc_values.append(0.0)
            
            physchem = np.array(rdkit_desc_values, dtype=np.float32)
            return np.concatenate([fp_arr, physchem])
        except:
            return None

    def predict(self, smiles):
        features = self._extract_features(smiles)
        if features is None:
            return "Invalid Structure"
        return round(float(self.model.predict([features])[0]), 2)
        
    def predict_batch(self, smiles_list):
        return [self.predict(s) for s in smiles_list]