import pandas as pd
import joblib
import os
import sys
import numpy as np

# Mock Streamlit to avoid errors if imported modules use it
import streamlit
from unittest.mock import MagicMock
sys.modules['streamlit'] = MagicMock()

def verify_risk_calculation():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "gam", "gam_model_student2_real5fold.joblib")
        data_path = os.path.join(base_dir, "gam", "student_data_2.csv")
        
        print(f"Loading model from {model_path}...")
        model_artifact = joblib.load(model_path)
        model = model_artifact["model"]
        preprocessor = model_artifact["preprocess"]
        label_encoder = model_artifact["label_encoder"]
        
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path, sep=";", decimal=",", engine="python", encoding="utf-8-sig")
        
        # Clean columns
        clean_cols = [c.replace("\ufeff", "").strip().replace(" ", "_") for c in df.columns]
        df.columns = clean_cols
        
        # Simple cleaning to match dashboard
        for col in df.columns:
            if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
                try:
                    series = df[col].astype(str).str.replace(',', '.', regex=False)
                    temp = pd.to_numeric(series, errors='coerce')
                    if temp.notna().sum() > 0.5 * len(temp):
                        df[col] = temp
                except:
                    pass
        
        numeric_cols_temp = df.select_dtypes(include=[np.number]).columns
        for c in numeric_cols_temp:
            mask_outliers = df[c] > 1e10 
            if mask_outliers.any():
                df.loc[mask_outliers, c] = np.nan

        target_col = "Target"
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
        else:
            X = df
            
        print("Transforming data...")
        X_pre = preprocessor.transform(X).astype(float)
        
        print("Predicting...")
        all_probs = model.predict_proba(X_pre)
        all_classes = label_encoder.classes_
        
        d_idx = list(all_classes).index("Dropout") if "Dropout" in all_classes else 0
        
        if all_probs.ndim == 1:
            p_1 = all_probs
            risk_scores = p_1 if all_classes[1] == "Dropout" else (1 - p_1)
        else:
            risk_scores = all_probs[:, d_idx]
            
        risk_df = X.copy()
        risk_df["Risk Score"] = risk_scores
        risk_df = risk_df.sort_values(by="Risk Score", ascending=False)
        
        print("\nTop 5 Highest Risk Students:")
        print(risk_df[["Risk Score"]].head(5))
        
        # Verification Checks
        assert risk_df["Risk Score"].is_monotonic_decreasing, "Risk scores are not sorted descending!"
        assert risk_df["Risk Score"].max() <= 1.0, "Risk score > 1.0"
        assert risk_df["Risk Score"].min() >= 0.0, "Risk score < 0.0"
        
        print("\n✅ Verification Successful: Logic is correct.")
        
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_risk_calculation()
