#!/usr/bin/env python3
"""
predict.py
Load pansirt.h5 (combined file produced by the QSAR pipeline) and predict for an input CSV (SMILES column).
This version applies saved scaler and feature_selector so the final feature shape matches training.
"""
import argparse
import os
import sys
import json
import base64
import pickle
import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from tensorflow.keras.models import load_model

def compute_full_feature_matrix(smiles_list):
    """
    Compute the same full feature block used in training:
    RDKit descriptors (Descriptors._descList) + 7 Lipinski fields + Morgan(1024) + MACCS (~167).
    Returns numpy array of shape (n_samples, n_full_features)
    """
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
        [x[0] for x in Descriptors._descList]
    )
    n_rdkit = len(descriptor_calculator.GetDescriptorNames())
    n_lip = 7
    n_morgan = 1024
    # MACCS is commonly 167 bits in many RDKit builds; training used whatever MACCS produced
    # We'll compute MACCS and handle length mismatches later
    descriptors_list = []
    morgan_list = []
    maccs_list = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            descriptors_list.append([np.nan] * n_rdkit + [np.nan]*n_lip)
            morgan_list.append([0]*n_morgan)
            maccs_list.append([0]*167)
            continue
        desc = descriptor_calculator.CalcDescriptors(mol)
        desc = [float(x) if x is not None else np.nan for x in desc]
        lip = [
            float(Lipinski.NumHDonors(mol)),
            float(Lipinski.NumHAcceptors(mol)),
            float(Lipinski.NumRotatableBonds(mol)),
            float(Lipinski.NumAromaticRings(mol)),
            float(Lipinski.NumAliphaticRings(mol)),
            float(Lipinski.NumSaturatedRings(mol)),
            float(Lipinski.NumHeteroatoms(mol))  # replace nonexistent NumHeterocycles
        ]
        descriptors_list.append(list(desc)+lip)
        morgan_list.append(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_morgan)))
        mac = list(MACCSkeys.GenMACCSKeys(mol))
        maccs_list.append(mac)

    desc_arr = np.array(descriptors_list, dtype=np.float64)
    morgan_arr = np.array(morgan_list, dtype=np.float64)
    maccs_arr = np.array(maccs_list, dtype=np.float64)

    # Ensure MACCS has consistent width (pad/truncate to 167)
    desired_maccs = 167
    if maccs_arr.shape[1] != desired_maccs:
        if maccs_arr.shape[1] < desired_maccs:
            pad = np.zeros((maccs_arr.shape[0], desired_maccs - maccs_arr.shape[1]))
            maccs_arr = np.hstack([maccs_arr, pad])
        else:
            maccs_arr = maccs_arr[:, :desired_maccs]

    full_X = np.hstack([desc_arr, morgan_arr, maccs_arr])
    return full_X

def load_combined_h5(h5path):
    out = {}
    hf = h5py.File(h5path, 'r')

    # keras model
    if 'keras_model_bytes' in hf:
        tmp = os.path.join(os.path.dirname(h5path), '_tmp_pansirt_keras.h5')
        arr = hf['keras_model_bytes'][()]
        with open(tmp, 'wb') as fh:
            fh.write(bytes(arr))
        out['keras_model_path'] = tmp
        try:
            out['keras_model'] = load_model(tmp)
        except Exception as e:
            print("Failed to load Keras model from combined file:", e)
            out['keras_model'] = None
    else:
        out['keras_model'] = None

    # sklearn models
    out['sk_models'] = {}
    if 'sklearn_models' in hf:
        grp = hf['sklearn_models']
        for key in grp.attrs:
            if key.endswith('_save_error'):
                continue
            b64 = grp.attrs[key]
            try:
                obj = pickle.loads(base64.b64decode(b64.encode('utf-8')))
                out['sk_models'][key] = obj
            except Exception as e:
                print(f"Failed to unpickle sklearn model {key}: {e}")

    # scaler
    out['scaler'] = None
    if 'scaler_pickle_b64' in hf.attrs:
        try:
            out['scaler'] = pickle.loads(base64.b64decode(hf.attrs['scaler_pickle_b64'].encode('utf-8')))
        except Exception:
            out['scaler'] = None

    # feature selector
    out['feature_selector'] = None
    if 'feature_selector_pickle_b64' in hf.attrs:
        try:
            out['feature_selector'] = pickle.loads(base64.b64decode(hf.attrs['feature_selector_pickle_b64'].encode('utf-8')))
        except Exception:
            out['feature_selector'] = None

    # feature_names (optional)
    out['feature_names'] = None
    if 'feature_names' in hf:
        try:
            out['feature_names'] = [x.decode('utf-8') for x in hf['feature_names'][()]]
        except Exception:
            out['feature_names'] = None

    # store expected feature_count (if selector exists, use selector.n_features_in_ or k)
    out['expected_feature_count'] = None
    if out['feature_selector'] is not None:
        try:
            # If the selector has attribute k, use it to compute output dim.
            k = getattr(out['feature_selector'], 'k', None)
            if k is not None:
                out['expected_feature_count'] = int(k)
            else:
                # If selector has get_support, we can infer number of True entries from training
                support = out['feature_selector'].get_support()
                out['expected_feature_count'] = int(np.sum(support))
        except Exception:
            out['expected_feature_count'] = None
    else:
        # No selector: try to infer from a saved sklearn model input dimension if possible
        for m in out['sk_models'].values():
            if hasattr(m, 'n_features_in_'):
                out['expected_feature_count'] = int(getattr(m, 'n_features_in_', None))
                break
        # keras model: inspect first dense layer expecting input_dim
        if out['expected_feature_count'] is None and out['keras_model'] is not None:
            try:
                # Keras's input shape info
                layer = out['keras_model'].layers[0]
                if getattr(layer, 'input_shape', None) is not None:
                    ish = layer.input_shape
                    # ish might be (None, input_dim)
                    if isinstance(ish, tuple):
                        out['expected_feature_count'] = int(ish[-1])
            except Exception:
                pass

    hf.close()
    return out

def main():
    parser = argparse.ArgumentParser(description='Use pansirt.h5 to predict pIC50 for a SMILES CSV')
    parser.add_argument('--models', default='pansirt.h5', help='Path to pansirt.h5')
    parser.add_argument('--input', required=True, help='Input CSV with SMILES column named SMILES')
    parser.add_argument('--output', default='predictions_from_pansirt.csv', help='Output CSV file')
    args = parser.parse_args()

    if not os.path.exists(args.models):
        print("Model file not found:", args.models)
        sys.exit(1)

    info = load_combined_h5(args.models)

    df = pd.read_csv(args.input)
    if 'SMILES' not in df.columns:
        print("Input CSV must have a 'SMILES' column")
        sys.exit(1)

    print("Computing descriptors/fingerprints for input SMILES...")
    X_full = compute_full_feature_matrix(df['SMILES'].tolist())
    print(f"Computed full feature matrix with shape: {X_full.shape}")

    # Replace NaNs/inf
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply scaler if present (scaler was fit on training X_train)
    X_scaled = X_full
    if info.get('scaler') is not None:
        try:
            X_scaled = info['scaler'].transform(X_full)
            print("Applied stored scaler to features.")
        except Exception as e:
            print("Stored scaler could not transform input (trying fit_transform fallback):", e)
            try:
                X_scaled = info['scaler'].fit_transform(X_full)
                print("Fallback: applied fit_transform of stored scaler.")
            except Exception as e2:
                print("Fallback scaler fit_transform failed:", e2)
                X_scaled = X_full
    else:
        print("No scaler found in pansirt.h5; using raw features.")

    # Apply feature selector if present (training applied scaler then selector)
    X_final = X_scaled
    if info.get('feature_selector') is not None:
        try:
            X_final = info['feature_selector'].transform(X_scaled)
            print(f"Applied feature selector; resulting shape: {X_final.shape}")
        except Exception as e:
            print("Feature selector transform failed:", e)
            # As a last resort, attempt to match expected_feature_count by truncating or padding
            expected = info.get('expected_feature_count')
            if expected is not None:
                print(f"Attempting to reshape features to expected_feature_count={expected}")
                n_samples = X_scaled.shape[0]
                n_cols = X_scaled.shape[1]
                if n_cols >= expected:
                    X_final = X_scaled[:, :expected]
                else:
                    pad = np.zeros((n_samples, expected - n_cols))
                    X_final = np.hstack([X_scaled, pad])
                print(f"Reshaped features to {X_final.shape}")
            else:
                print("No expected feature count available; continuing with scaled full features.")
                X_final = X_scaled
    else:
        # No selector: attempt to ensure X_final columns match expected_feature_count (if any)
        expected = info.get('expected_feature_count')
        if expected is not None:
            if X_scaled.shape[1] != expected:
                print(f"Warning: computed features have {X_scaled.shape[1]} cols but models expect {expected}. Trying to adjust by truncating/padding.")
                n_samples = X_scaled.shape[0]
                if X_scaled.shape[1] >= expected:
                    X_final = X_scaled[:, :expected]
                else:
                    pad = np.zeros((n_samples, expected - X_scaled.shape[1]))
                    X_final = np.hstack([X_scaled, pad])
                print(f"Adjusted features to shape: {X_final.shape}")
            else:
                X_final = X_scaled
        else:
            X_final = X_scaled

    # Now predict using sklearn models and keras model in info
    results = {}
    for name, model in info.get('sk_models', {}).items():
        try:
            preds = model.predict(X_final)
            results[name] = preds
            print(f"Predictions computed for sklearn model: {name}")
        except Exception as e:
            print(f"Failed to predict with {name}: {e}")

    if info.get('keras_model') is not None:
        try:
            kmod = info['keras_model']
            kp = kmod.predict(X_final)
            kp = np.array(kp)
            if kp.ndim == 2 and kp.shape[1] == 1:
                kp = kp.flatten()
            results['DeepLearning'] = kp
            print("Predictions computed for Keras model (DeepLearning).")
        except Exception as e:
            print("Failed to predict with Keras model:", e)

    # Clean up tmp keras file if present
    tmp_path = info.get('keras_model_path', None)
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # Assemble output
    out_df = df.copy()
    for name, arr in results.items():
        if isinstance(arr, np.ndarray) and arr.ndim > 1:
            for j in range(arr.shape[1]):
                out_df[f'Pred_{name}_{j}'] = arr[:, j]
        else:
            out_df[f'Pred_{name}'] = arr

    out_df.to_csv(args.output, index=False)
    print("Saved predictions to", args.output)

if __name__ == '__main__':
    main()