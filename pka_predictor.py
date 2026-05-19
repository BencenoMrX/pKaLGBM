# pka_predictor class
import os
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator

class PkaPredictor:
    def __init__(self, model_path='pka_lightgbm_model.pkl'):
        # Loads model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Initialize the Morgan generator once, using new Morgan rdkit code
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        
    def _extract_features(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            
            # 1024-bit Morgan fingerprint
            fp = self.morgan_gen.GetFingerprint(mol)
            fp_arr = np.zeros((1,), dtype=np.float32)
            Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
            
            # 2. Gasteiger charges (partial atomic charges)
            mol_h = Chem.AddHs(mol)
            AllChem.ComputeGasteigerCharges(mol_h)
            charges = [float(a.GetProp('_GasteigerCharge')) for a in mol_h.GetAtoms() 
                       if a.HasProp('_GasteigerCharge') and not np.isnan(float(a.GetProp('_GasteigerCharge')))]
            max_charge = max(charges) if charges else 0.0
            min_charge = min(charges) if charges else 0.0
            
            # RDKit Descriptors
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
            return "Invalid structure"
        # Ignore the feature name warning during prediction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            prediction = self.model.predict([features])[0]
            
        return round(float(self.model.predict([features])[0]), 2)
        
    def predict_batch(self, smiles_list):
        return [self.predict(s) for s in smiles_list]
