import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys
import subprocess

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from explainability import calculate_shap_values, generate_genai_explanation

# --- Mappings ---
application_mode_map = {
    1: '1st phase - general contingent',
    2: 'Ordinance No. 612/93',
    3: '1st phase - special contingent (Azores Island)',
    4: 'Holders of other higher courses',
    5: 'Ordinance No. 854-B/99',
    6: 'International student (bachelor)',
    7: '1st phase - special contingent (Madeira Island)',
    8: '2nd phase - general contingent',
    9: '3rd phase - general contingent',
    10: 'Ordinance No. 533-A/99 (Different Plan)',
    11: 'Ordinance No. 533-A/99 (Other Institution)',
    12: 'Over 23 years old',
    13: 'Transfer',
    14: 'Change of course',
    15: 'Technological specialization diploma holders',
    16: 'Change of institution/course',
    17: 'Short cycle diploma holders',
    18: 'Change of institution/course (International)'
}

course_map = {
    1: 'Biofuel Production Technologies',
    2: 'Animation and Multimedia Design', 
    3: 'Social Service (evening attendance)',
    4: 'Agronomy',
    5: 'Communication Design',
    6: 'Veterinary Nursing',
    7: 'Informatics Engineering',
    8: 'Equinculture',
    9: 'Management',
    10: 'Social Service',
    11: 'Tourism',
    12: 'Nursing',
    13: 'Oral Hygiene',
    14: 'Advertising and Marketing Management',
    15: 'Journalism and Communication',
    16: 'Basic Education',
    17: 'Management (evening attendance)'
}

# Page Config
st.set_page_config(page_title="Student Success Dashboard (v2 - 5Fold)", layout="wide")

# Title
st.title("🎓 Student Success & Dropout Risk Dashboard (v2.1 - Tracker Active)")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# -------------------------------
# AI Threshold Recommendation Modal (Session State)
# -------------------------------
if "show_threshold_modal" not in st.session_state:
    st.session_state.show_threshold_modal = False

if "high_risk_threshold" not in st.session_state:
    st.session_state.high_risk_threshold = 0.7

if "low_risk_threshold" not in st.session_state:
    st.session_state.low_risk_threshold = 0.3

# Sidebar button to open modal
if st.sidebar.button("💡 AI Threshold Recommendation"):
    st.session_state.show_threshold_modal = True


# Load Data & Model
@st.cache_resource
def load_resources():
    # Paths - Adjusted for GAM model
    # Expecting dashboard.py to be in predictive_model/, so gam/ is a subdir
    model_path = os.path.join(os.path.dirname(__file__), "gam", "gam_model_student2_real5fold.joblib")
    data_path = os.path.join(os.path.dirname(__file__), "gam", "student_data_2.csv")
    
    # Load
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please run `predictive_model/gam/gam_real_5fold.py` first.")
        return None, None
        
    # The GAM model is saved as a dictionary: {"model": gam, "preprocess": pre, "label_encoder": label_enc}
    try:
        model_artifact = joblib.load(model_path)
    except Exception as e:
        # Auto-Recovery Logic
        st.warning(f"Model load failed ({e}). Attempting to auto-fix for your device...")
        
        try:
            # Locate the fix script
            fix_script = os.path.join(os.path.dirname(__file__), "gam", "gam_real_5fold.py")
            if not os.path.exists(fix_script):
                 st.error(f"Cannot find fix script at {fix_script}")
                 return None, None
            
            # Run it
            st.info("Retraining model with gam_real_5fold.py... This may take a moment.")
            process = subprocess.run([sys.executable, fix_script], capture_output=True, text=True)
            if process.returncode != 0:
                st.error(f"Auto-fix failed:\n{process.stderr}")
                return None, None
                
            st.success("Model optimized! Reloading...")
            model_artifact = joblib.load(model_path)
            
        except Exception as e2:
             st.error(f"Fatal error loading model: {e2}")
             return None, None

    if not isinstance(model_artifact, dict):
        st.error("Model file format unexpected. Expected a dictionary from gam_real_5fold.py.")
        return None, None

    # Load CSV with robust handling similar to gam_fix.py
    if not os.path.exists(data_path):
        st.error(f"Data not found at {data_path}")
        return None, None

    df = pd.read_csv(data_path, sep=";", decimal=",", engine="python", encoding="utf-8-sig")
    
    # Clean columns
    clean_cols = [c.replace("\ufeff", "").strip().replace(" ", "_") for c in df.columns]
    df.columns = clean_cols
    
    # --- Robust Numeric Cleaning (Mirroring gam_fix.py) ---
    # The dashboard needs to clean raw data exactly like the training script
    # otherwise the Imputer will fail on string values like '1,34286E+16'
    
    # 1. Identify and fix string-numeric columns
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            try:
                # Replace comma with dot and coerce to numeric
                series = df[col].astype(str).str.replace(',', '.', regex=False)
                temp = pd.to_numeric(series, errors='coerce')
                
                # Check if it looks numeric (heuristic: >50% valid)
                valid_count = temp.notna().sum()
                if valid_count > 0.5 * len(temp):
                    df[col] = temp
            except:
                pass

    # 2. Clean extreme outliers (>1e10)
    # Correcting data issue where decimal point is missing (e.g. 1.34e16 instead of 13.4)
    numeric_cols_temp = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols_temp:
        mask_outliers = df[c] > 1e10
        if mask_outliers.any():
            # Assume these are shifted by 1e15 (16-17 digits -> xx.xxxxx)
            df.loc[mask_outliers, c] = df.loc[mask_outliers, c] / 1e15
    
    # 3. Robust Fix for Grades specifically (ensure range 0-20)
    # This handles cases like 13875 (should be 13.875) which are not > 1e10
    grade_cols = [c for c in df.columns if 'grade' in c.lower()]
    for c in grade_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
             # Apply heuristic: recursive division by 10 until <= 20
             def fix_grade_scale(x):
                 if pd.isna(x): return x
                 while x > 20.0:
                     x /= 10.0
                 return x
             
             df[c] = df[c].apply(fix_grade_scale)
            
    return model_artifact, df

def clean_feature_name(name):
    """Cleans technical feature names (e.g. cat__Course_12) into readable text."""
    # Remove Sklearn/PyGAM prefixes
    name = name.replace("cat__", "").replace("num__", "")
    
    # Check for One-Hot suffix (usually _<number>)
    # But be careful with names that naturally end in numbers like "sem_1"
    # Usually one-hot encoding appends an underscore and the value.
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        base = parts[0].replace("_", " ")
        val = parts[1]
        
        # Optional: Try to map specific columns if known
        if "Course" in base:
            val = course_map.get(int(val), val)
        elif "Application mode" in base:
             val = application_mode_map.get(int(val), val)
             
        return f"{base}: {val}"
        
    return name.replace("_", " ")

model_artifact, df = load_resources()

if model_artifact is not None and df is not None:
    model = model_artifact["model"]
    preprocessor = model_artifact["preprocess"]
    label_encoder = model_artifact["label_encoder"]

    # --- TRACKING DATA HANDLING ---
    tracking_file = os.path.join(os.path.dirname(__file__), "tracking_data.csv")
    
    def load_tracking_data():
        if os.path.exists(tracking_file):
            # Load with index as is, then ensure type consistency
            # We treat index as int (Student_Index)
            try:
                td = pd.read_csv(tracking_file, index_col="Student_Index")
                return td
            except Exception:
                return pd.DataFrame(columns=["Is_Tracked", "Notes"])
        else:
            return pd.DataFrame(columns=["Is_Tracked", "Notes"])

    def save_tracking_data(index, is_tracked, notes):
        # Load current
        td = load_tracking_data()
        
        # Update
        td.loc[index, "Is_Tracked"] = 1 if is_tracked else 0
        td.loc[index, "Notes"] = notes
        
        td.index.name = "Student_Index"
        td.to_csv(tracking_file)
        return td

    # Load initially
    tracking_df = load_tracking_data()

    # Target Logic
    target_col = "Target"
    known_targets = ["Output", "Target", "Status", "Outcome"]
    for t in known_targets:
        if t in df.columns:
            target_col = t
            break
            
    if target_col not in df.columns:
         st.error(f"Could not find target column. Available columns: {list(df.columns)}")
         st.stop()
    
    X = df.drop(columns=[target_col])
    # We don't perform extensive cleaning here because the preprocessor in the artifact handles it
    # BUT we need to ensure formatting matches what preprocessor expects (e.g. comma decimals if raw)
    # pd.read_csv handled decimal=',' above, so dtypes should be mostly correct.
    
    # --- Sidebar Filters ---
    st.sidebar.subheader("Filter Students")
    
    # Threshold Sliders
    st.session_state.high_risk_threshold = st.sidebar.slider(
        "High Risk Threshold", 0.0, 1.0, 
        st.session_state.high_risk_threshold, 0.05
    )
    
    st.session_state.low_risk_threshold = st.sidebar.slider(
        "Safe Threshold", 0.0, 1.0, 
        st.session_state.low_risk_threshold, 0.05
    )



    # --- Sidebar Filters ---
    
    # Data Source Selection (Logic Split)
    # If "Simulate", we skip the list selection logic.
    # If "Existing", we use the list list selection.
    
    # Data Source Selection
    data_source = st.sidebar.radio("Data Source", ["Select Existing Student", "Simulate New Student"])
    
    selected_student_data = None
    selected_student_index = None

    if data_source == "Select Existing Student":
        
        # --- 1. Global Risk Calculation & List ---
        with st.spinner("Analyzing all students..."):
            try:
                # Transform & Predict for ALL
                X_pre = preprocessor.transform(X).astype(float)
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
                
                # --- NEW: Group Filters ---
                st.sidebar.markdown("---")
                st.sidebar.subheader("Group Filters")
                
                # Course Filter
                # Ensure we have the map accessible. course_map is global.
                available_courses = sorted(list(course_map.values()))
                sel_course_label = st.sidebar.selectbox("Filter by Course", ["All"] + available_courses)
                
                # App Mode Filter
                available_app_modes = sorted(list(application_mode_map.values()))
                sel_app_mode_label = st.sidebar.selectbox("Filter by Mode", ["All"] + available_app_modes)
                
                # Apply Filters
                filtered_risk_df = risk_df.copy()
                
                if sel_course_label != "All":
                    # Map label back to ID
                    c_rev = {v: k for k, v in course_map.items()}
                    c_id = c_rev.get(sel_course_label)
                    if c_id:
                        filtered_risk_df = filtered_risk_df[filtered_risk_df["Course"] == c_id]
                        
                if sel_app_mode_label != "All":
                    am_rev = {v: k for k, v in application_mode_map.items()}
                    am_id = am_rev.get(sel_app_mode_label)
                    if am_id:
                         # Handle potentially different column names
                         col_name = "Application_mode" if "Application_mode" in risk_df.columns else "Application mode"
                         filtered_risk_df = filtered_risk_df[filtered_risk_df[col_name] == am_id]

                # --- NEW: Dynamic Attribute Filters ---
                st.sidebar.markdown("---")
                with st.sidebar.expander("🛠️ Advanced Filters"):
                    # Exclude 'Risk Score' and 'Target'
                    cols_to_exclude = ["Risk Score"]
                    if "target_col" in locals():
                        cols_to_exclude.append(target_col)
                        
                    # Get available columns based on original df logic (but using risk_df for current state)
                    available_attribs = sorted([c for c in risk_df.columns if c not in cols_to_exclude])
                    
                    selected_attribs = st.multiselect(
                        "Add Custom Filter", 
                        options=available_attribs,
                        key="dynamic_filter_multiselect"
                    )
                    
                    for attr in selected_attribs:
                        # Check type
                        col_data = risk_df[attr]
                        is_numeric = pd.api.types.is_numeric_dtype(col_data)
                        
                        # Heuristic: if many unique numeric values, use slider. If few, use select.
                        unique_count = col_data.nunique()
                        
                        if is_numeric and unique_count > 10:
                            min_val = float(col_data.min())
                            max_val = float(col_data.max())
                            
                            if min_val == max_val:
                                st.caption(f"{attr}: Constant value {min_val}")
                            else:
                                rng = st.slider(
                                    f"{attr}", 
                                    min_value=min_val, 
                                    max_value=max_val, 
                                    value=(min_val, max_val),
                                    key=f"dyn_slider_{attr}"
                                )
                                filtered_risk_df = filtered_risk_df[
                                    (filtered_risk_df[attr] >= rng[0]) & 
                                    (filtered_risk_df[attr] <= rng[1])
                                ]
                        else:
                            # Categorical or Low-Cardinality Numeric
                            # Convert to string for display consistency
                            unique_vals = sorted(col_data.dropna().unique())
                            unique_vals_str = [str(x) for x in unique_vals]
                            
                            sel_vals = st.multiselect(
                                f"{attr}",
                                options=unique_vals_str,
                                default=unique_vals_str,
                                key=f"dyn_multi_{attr}"
                            )
                            
                            # Filter logic
                            if len(sel_vals) < len(unique_vals_str):
                                mask = col_data.astype(str).isin(sel_vals)
                                filtered_risk_df = filtered_risk_df[mask]

                # --- NEW: Group Summary ---
                st.subheader("📊 Group Summary")
                
                # Metrics
                total_students = len(filtered_risk_df)
                
                # High Risk
                high_risk_students = filtered_risk_df[filtered_risk_df["Risk Score"] > st.session_state.high_risk_threshold]
                high_risk_count = len(high_risk_students)
                
                # Likely Graduate (Safe)
                safe_students = filtered_risk_df[filtered_risk_df["Risk Score"] <= st.session_state.low_risk_threshold]
                safe_count = len(safe_students)
                
                avg_risk = filtered_risk_df["Risk Score"].mean() if total_students > 0 else 0.0
                
                # Disclaimer
                if 0 < total_students < 30:
                    st.warning(f"⚠️ Small Sample Size: This group has only {total_students} students. Statistical insights may be limited.")
                elif total_students == 0:
                    st.warning("No students match the selected filters.")
                    
                # Display Metrics and Intervention in Columns
                gs_col1, gs_col2 = st.columns([2, 1])
                
                with gs_col1:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Students", total_students)
                    m2.metric("High Risk Students", high_risk_count, delta_color="inverse")
                    m3.metric("Students Likely to Graduate", safe_count, delta_color="normal") # Green is good
                    m4.metric("Avg Risk", f"{avg_risk:.1%}")
                
                with gs_col2:
                    # Bulk Intervention
                    with st.expander("📢 Bulk Actions"):
                        # Target Selection
                        target_group = st.radio("Target Group", ["High Risk", "Likely Graduates"], horizontal=True, label_visibility="collapsed")
                        
                        if target_group == "High Risk":
                            if high_risk_count > 0:
                                st.write(f"**Target:** {high_risk_count} Risk Students")
                                action_type = st.selectbox("Action", ["Send Email", "Schedule Meeting", "Notify Tutors"], key="bulk_action_risk")
                                
                                if st.button(f"Execute Action"):
                                     st.success(f"'{action_type}' queued for {high_risk_count} students.")
                            else:
                                st.info("No high risk students found.")
                                
                        else: # Likely Graduates
                            if safe_count > 0:
                                st.write(f"**Target:** {safe_count} Safe Students")
                                action_type = st.selectbox("Action", ["Send Kudos", "Invite to Honor Society", "Ask for Testimonial"], key="bulk_action_safe")
                                
                                if st.button(f"Execute Action"):
                                     st.balloons() # Fun effect for kudos
                                     st.success(f"'{action_type}' queued for {safe_count} students!")
                            else:
                                st.info("No likely graduates found.")
                        
                st.markdown("---")

                # Update risk_df to be the filtered version for the rest of the UI (List & Selection)
                risk_df = filtered_risk_df
                
                # --- Sidebar Sync Logic ---
                # Check if current selection is still valid
                if "selected_student_idx" not in st.session_state:
                     st.session_state.selected_student_idx = risk_df.index[0] if not risk_df.empty else None
                elif not risk_df.empty and st.session_state.selected_student_idx not in risk_df.index:
                     st.session_state.selected_student_idx = risk_df.index[0]

                # Helper
                def get_index_pos(val, options):
                    try: return list(options).index(val) 
                    except: return 0

                # Sidebar Selectbox
                sb_options = risk_df.index if not risk_df.empty else []
                sb_student = st.sidebar.selectbox(
                    "Select Student Index", 
                    options=sb_options,
                    index=get_index_pos(st.session_state.selected_student_idx, sb_options),
                    key="sb_student_select"
                )
                
                # Update state from sidebar immediately if it changed
                if sb_student is not None and sb_student != st.session_state.selected_student_idx:
                    st.session_state.selected_student_idx = sb_student
                    # No rerun needed here, the flow continues with the new value
                
                if risk_df.empty:
                    st.error("No students found with current filters.")
                    st.stop()

                # --- List Display ---
                st.subheader("📋 Student Risk Overview")
                
                # --- MERGE TRACKING DATA ---
                # We want "Is Tracked" (0/1) and "Tracking Action" (Text)
                
                # 1. Join tracking data
                # Ensure indices match types (heuristic: convert to common type if needed, but usually Int64)
                # Left join risk_df with tracking_df
                
                merged_df = risk_df.join(tracking_df, how="left")
                
                # 2. Process columns
                merged_df["Is Tracked"] = merged_df["Is_Tracked"].fillna(0).astype(int)
                merged_df["Tracking Action"] = merged_df["Notes"].fillna("")
                
                # --- Column Visibility ---
                all_cols = list(merged_df.columns)
                # Default: Risk Score first, then Tracked info, then rest
                # Filter out raw 'Is_Tracked' and 'Notes' from display if they exist
                cols_to_hide_raw = ["Is_Tracked", "Notes"]
                display_candidates = [c for c in all_cols if c not in cols_to_hide_raw]
                
                default_cols = ["Risk Score", "Is Tracked", "Tracking Action"] + [c for c in display_candidates if c not in ["Risk Score", "Is Tracked", "Tracking Action"]]
                
                with st.expander("👁️ Show/Hide Columns"):
                    display_cols = st.multiselect("Select Columns to Display", options=display_candidates, default=default_cols)
                
                if not display_cols: 
                    st.warning("Please select at least one column to display.")
                    display_cols = ["Risk Score"]

                def color_rows(row):
                    # Default Risk Colors
                    val = row["Risk Score"]
                    styles = []
                    
                    # Highlight tracked rows
                    bg_color = ''
                    if row["Is Tracked"] == 1:
                         bg_color = 'background-color: #E6F3FF;' # Light Blue
                    
                    # Risk Text Color
                    risk_color = 'red' if val > st.session_state.high_risk_threshold else ('green' if val < st.session_state.low_risk_threshold else 'orange')
                    
                    for col in row.index:
                        style = bg_color
                        if col == "Risk Score":
                            style += f' color: {risk_color}; font-weight: bold;'
                        styles.append(style)
                        
                    return styles

                # Display with Styler
                event = st.dataframe(
                    merged_df[display_cols].style
                    .format({"Risk Score": "{:.1%}"})
                    .apply(color_rows, axis=1),
                    height=300,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="risk_list_table" 
                )
                
                # Handle List Selection
                if len(event.selection.rows) > 0:
                    selected_row_idx = event.selection.rows[0]
                    list_selected_idx = risk_df.index[selected_row_idx]
                    
                    # If list click is different from current state, update state
                    if list_selected_idx != st.session_state.selected_student_idx:
                         st.session_state.selected_student_idx = list_selected_idx
                         st.rerun() # Rerun to sync sidebar

                # --- Valid Selection? ---
                selected_student_index = st.session_state.selected_student_idx
                selected_student_data = X.loc[[selected_student_index]]
                
            except Exception as e:
                st.error(f"Global analysis/selection failed: {e}")
                st.stop()
        

    
    else:
        # --- Simulate New Student ---
        st.sidebar.markdown("### Simulate Student")
        selected_student_index = "Simulated_User"
        
        # --- INPUT FORM ---
        # 1. Application Mode (Reverse Map for UI)
        # Create reverse map: "Name": ID
        app_mode_rev = {v: k for k, v in application_mode_map.items()}
        # Add any missing values from X that might not be in the map (just in case)
        # valid_app_modes = sorted(list(app_mode_rev.keys()))
        s_app_mode_label = st.sidebar.selectbox("Application Mode", options=list(app_mode_rev.keys()))
        s_app_mode = app_mode_rev[s_app_mode_label]

        # 2. Course
        course_rev = {v: k for k, v in course_map.items()}
        # Add generic options for ids 1-17 if map is incomplete, but we fixed it.
        s_course_label = st.sidebar.selectbox("Course", options=list(course_rev.keys()))
        s_course = course_rev[s_course_label]

        # 3. Tuition Fees
        s_tuition = st.sidebar.selectbox("Tuition fees up to date?", ["Yes", "No"])
        s_tuition_val = 1 if s_tuition == "Yes" else 0

        # 4. Age
        s_age = st.sidebar.number_input("Age at enrollment", min_value=17, max_value=70, value=20)

        # 5. Grades & Units (1st Sem)
        st.sidebar.markdown("#### 1st Semester")
        s_u1_enrolled = st.sidebar.number_input("Units Enrolled (1st)", 0, 20, 5)
        s_u1_approved = st.sidebar.number_input("Units Approved (1st)", 0, 20, 5)
        s_u1_grade = st.sidebar.number_input("Grade Avg (1st)", 0.0, 20.0, 14.0)

        # 6. Grades & Units (2nd Sem)
        st.sidebar.markdown("#### 2nd Semester")
        s_u2_enrolled = st.sidebar.number_input("Units Enrolled (2nd)", 0, 20, 5)
        s_u2_approved = st.sidebar.number_input("Units Approved (2nd)", 0, 20, 5)
        s_u2_grade = st.sidebar.number_input("Grade Avg (2nd)", 0.0, 20.0, 14.0)

        # Construct DataFrame with EXACT columns as X
        sim_data = {
            'Application_mode': [s_app_mode],
            'Course': [s_course],
            'Tuition_fees_up_to_date': [s_tuition_val],
            'Age_at_enrollment': [s_age],
            'Curricular_units_1st_sem_(enrolled)': [s_u1_enrolled],
            'Curricular_units_1st_sem_(approved)': [s_u1_approved],
            'Curricular_units_1st_sem_(grade)': [s_u1_grade],
            'Curricular_units_2nd_sem_(enrolled)': [s_u2_enrolled],
            'Curricular_units_2nd_sem_(approved)': [s_u2_approved],
            'Curricular_units_2nd_sem_(grade)': [s_u2_grade]
        }
        selected_student_data = pd.DataFrame(sim_data) # This will be used by the rest of the app
    
    # --- Prediction Logic ---
    # Predict
    # GAM needs preprocessed data
    classes = None  # Safe initialization
    probs = None

    try:
        student_data_pre = preprocessor.transform(selected_student_data).astype(float)
        probs = model.predict_proba(student_data_pre)
        classes = label_encoder.classes_
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()
    
    if classes is None or probs is None:
        st.error("Model prediction failed. Check logs.")
        st.stop()
    
    # Find index of "Dropout" class
    dropout_idx = list(classes).index("Dropout") if "Dropout" in classes else 0
    # Handle prob shape (n_samples, n_classes) or (n_samples,) if binary
    if probs.ndim == 1:
        # Binary case for pygam often returns just P(y=1)
        # We need to check label encoder to see which is 1
        # Typically 1 is the second class in classes_
        p_1 = probs[0]
        dropout_prob = p_1 if classes[1] == "Dropout" else (1 - p_1)
    else:
        dropout_prob = probs[0][dropout_idx]
    
    # Display Status (Sidebar) -- RESTORED
    st.sidebar.markdown("### Prediction Status")
    if dropout_prob >= st.session_state.high_risk_threshold:
        st.sidebar.error(f"🚨 HIGH RISK ({dropout_prob:.1%})")
    elif dropout_prob <= st.session_state.low_risk_threshold:
        st.sidebar.success(f"✅ SAFE ({dropout_prob:.1%})")
    else:
        st.sidebar.warning(f"⚠️ MONITOR ({dropout_prob:.1%})")
            
    # --- Main Content Details ---
    
    st.markdown("---") # Separator between list and details
    
    # --- TRACKING & ACTIONS ---
    if data_source == "Select Existing Student":
        with st.expander("📝 Actions & Tracking", expanded=True):
            # Check current status
            curr_tracked_val = 0
            curr_notes = ""
            
            if selected_student_index in tracking_df.index:
                curr_tracked_val = int(tracking_df.loc[selected_student_index, "Is_Tracked"])
                curr_notes = str(tracking_df.loc[selected_student_index, "Notes"])
                if curr_notes == "nan": curr_notes = ""
            
            curr_tracked = (curr_tracked_val == 1)

            # Callback to save immediately
            def on_track_change():
                save_tracking_data(
                    selected_student_index, 
                    st.session_state.track_checkbox, 
                    st.session_state.track_notes
                )
                
            # UI Components
            c1, c2 = st.columns([1, 3])
            with c1:
                is_tracked_input = st.checkbox(
                    "Mark as Tracked", 
                    value=curr_tracked,
                    key="track_checkbox",
                    on_change=on_track_change
                )
                if is_tracked_input:
                    st.caption("✅ Saved to 'Is Tracked'")
            
            with c2:
                notes_input = st.text_area(
                    "Action Notes / Tracking Action", 
                    value=curr_notes,
                    placeholder="e.g. Sent academic warning email on 12/05...",
                    key="track_notes",
                    on_change=on_track_change
                )

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Student Profile (ID: {selected_student_index})")

        # --- Representative Data Check ---
        sel_course = selected_student_data["Course"].iloc[0] if "Course" in selected_student_data.columns else None
        
        app_mode_col = "Application_mode" if "Application_mode" in selected_student_data.columns else ("Application mode" if "Application mode" in selected_student_data.columns else None)
        sel_app_mode = selected_student_data[app_mode_col].iloc[0] if app_mode_col else None

        if sel_course is not None and data_source != "Simulate New Student":
            # Only check existing data counts if we have X loaded
             course_count = X[X["Course"] == sel_course].shape[0] if 'X' in locals() else 50
             if course_count < 50:
                 st.warning(f"⚠️ Low Sample Size: This Course has only {course_count} students. Predictions may be less reliable.")

        # Create a display copy to show readable labels
        display_data = selected_student_data.copy()
    
        # Helper to safely get value
        def get_val(col):
            return display_data[col].iloc[0] if col in display_data.columns else "N/A"

        # Map values for display
        d_app_mode = get_val("Application_mode") if "Application_mode" in display_data.columns else get_val("Application mode")
        if str(d_app_mode).replace('.','').isdigit():
             d_app_mode = application_mode_map.get(int(d_app_mode), d_app_mode)
             
        d_course = get_val("Course")
        if str(d_course).replace('.','').isdigit():
             d_course = course_map.get(int(d_course), d_course)

        # --- NEW CARD LAYOUT ---
        # Row 1: Key Identifiers
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown(f"<h4 style='color: #4A90E2;'>Course</h4>", unsafe_allow_html=True)
            st.markdown(f"{d_course}")
        with r1c2:
            st.markdown(f"<h4 style='color: #4A90E2;'>Application Mode</h4>", unsafe_allow_html=True)
            st.markdown(f"{d_app_mode}")
            
        # Row 2: Demographics & Fees
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown(f"<h4 style='color: #4A90E2;'>Age at Enrollment</h4>", unsafe_allow_html=True)
            age_val = int(get_val("Age_at_enrollment")) if get_val("Age_at_enrollment") != "N/A" else "N/A"
            st.markdown(f"{age_val}")
        with r2c2:
            st.markdown(f"<h4 style='color: #4A90E2;'>Tuition Fees</h4>", unsafe_allow_html=True)
            t_status = "Up to Date" if get_val("Tuition_fees_up_to_date") == 1 else "Overdue"
            st.markdown(f"{t_status}")

        # Row 3: Academic Performance (1st Sem)
        st.markdown(f"<h5 style='color: #17A589;'>1st Semester Performance</h5>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<h4 style='color: #4A90E2;'>Grade Avg</h4>", unsafe_allow_html=True)
            val = f"{get_val('Curricular_units_1st_sem_(grade)'):.2f}" if get_val('Curricular_units_1st_sem_(grade)') != "N/A" else "N/A"
            st.markdown(val)
        with m2:
            st.markdown(f"<h4 style='color: #4A90E2;'>Enrolled</h4>", unsafe_allow_html=True)
            val = int(get_val('Curricular_units_1st_sem_(enrolled)')) if get_val('Curricular_units_1st_sem_(enrolled)') != "N/A" else "N/A"
            st.markdown(val)
        with m3:
            st.markdown(f"<h4 style='color: #4A90E2;'>Approved</h4>", unsafe_allow_html=True)
            val = int(get_val('Curricular_units_1st_sem_(approved)')) if get_val('Curricular_units_1st_sem_(approved)') != "N/A" else "N/A"
            st.markdown(val)

        # Row 4: Academic Performance (2nd Sem)
        st.markdown(f"<h5 style='color: #17A589;'>2nd Semester Performance</h5>", unsafe_allow_html=True)
        n1, n2, n3 = st.columns(3)

        with n1:
             st.markdown(f"<h4 style='color: #4A90E2;'>Grade Avg</h4>", unsafe_allow_html=True)
             val = f"{get_val('Curricular_units_2nd_sem_(grade)'):.2f}" if get_val('Curricular_units_2nd_sem_(grade)') != "N/A" else "N/A"
             st.markdown(val)
        with n2:
             st.markdown(f"<h4 style='color: #4A90E2;'>Enrolled</h4>", unsafe_allow_html=True)
             val = int(get_val('Curricular_units_2nd_sem_(enrolled)')) if get_val('Curricular_units_2nd_sem_(enrolled)') != "N/A" else "N/A"
             st.markdown(val)
        with n3:
             st.markdown(f"<h4 style='color: #4A90E2;'>Approved</h4>", unsafe_allow_html=True)
             val = int(get_val('Curricular_units_2nd_sem_(approved)')) if get_val('Curricular_units_2nd_sem_(approved)') != "N/A" else "N/A"
             st.markdown(val)
        
        st.markdown("---")

        st.markdown("---")

        # --- Ceteris Paribus / What-If Analysis ---
        with st.expander("🛠️ Ceteris Paribus Analysis (What-If?)"):
            st.markdown("Simulate changes to this student's profile to see how the risk changes.")
            
            # Form for simulation
            with st.form("what_if_form"):
                cp_col1, cp_col2, cp_col3 = st.columns(3)
                
                with cp_col1:
                    # Binary / Categorical features
                    cp_tuition = st.checkbox("Tuition fees up to date", value=(get_val("Tuition_fees_up_to_date") == 1))
                    
                    # Application Mode
                    cp_current_app_mode = get_val("Application_mode") if "Application_mode" in display_data.columns else get_val("Application mode")
                    if str(cp_current_app_mode).replace('.','').isdigit():
                         cp_current_app_mode_label = application_mode_map.get(int(cp_current_app_mode), "Unknown")
                    else:
                         cp_current_app_mode_label = cp_current_app_mode
                         
                    # Reverse map for selection (safely)
                    app_mode_options = list(application_mode_map.values())
                    try:
                        app_mode_index = app_mode_options.index(cp_current_app_mode_label)
                    except:
                        app_mode_index = 0
                        
                    cp_app_mode_label = st.selectbox("Application Mode", options=app_mode_options, index=app_mode_index)

                with cp_col2:
                    # Numeric - Grades
                    cp_grade1 = st.number_input("1st Sem Grade", min_value=0.0, max_value=20.0, value=float(get_val('Curricular_units_1st_sem_(grade)') if get_val('Curricular_units_1st_sem_(grade)') != "N/A" else 0.0))
                    cp_grade2 = st.number_input("2nd Sem Grade", min_value=0.0, max_value=20.0, value=float(get_val('Curricular_units_2nd_sem_(grade)') if get_val('Curricular_units_2nd_sem_(grade)') != "N/A" else 0.0))

                with cp_col3:
                     # Numeric - Units Approved
                    cp_app1 = st.number_input("1st Sem Approved", min_value=0, max_value=30, value=int(get_val('Curricular_units_1st_sem_(approved)') if get_val('Curricular_units_1st_sem_(approved)') != "N/A" else 0))
                    cp_app2 = st.number_input("2nd Sem Approved", min_value=0, max_value=30, value=int(get_val('Curricular_units_2nd_sem_(approved)') if get_val('Curricular_units_2nd_sem_(approved)') != "N/A" else 0))
                
                submitted = st.form_submit_button("Simulate New Risk")
                
                if submitted:
                    # 1. Create modified dataframe
                    modified_student = selected_student_data.copy()
                    
                    # Update values
                    modified_student["Tuition_fees_up_to_date"] = 1 if cp_tuition else 0
                    
                    # Map back App Mode label to ID
                    app_mode_rev_map = {v: k for k, v in application_mode_map.items()}
                    modified_student["Application_mode"] = app_mode_rev_map.get(cp_app_mode_label, 1) # Default to 1 if fail
                    
                    modified_student["Curricular_units_1st_sem_(grade)"] = cp_grade1
                    modified_student["Curricular_units_2nd_sem_(grade)"] = cp_grade2
                    modified_student["Curricular_units_1st_sem_(approved)"] = cp_app1
                    modified_student["Curricular_units_2nd_sem_(approved)"] = cp_app2
                    
                    # 2. Predict
                    try:
                        ms_pre = preprocessor.transform(modified_student).astype(float)
                        ms_probs = model.predict_proba(ms_pre)
                        
                        # Calculate Risk (Dropout Prob)
                        if ms_probs.ndim == 1:
                             new_risk = ms_probs[0] if classes[1] == "Dropout" else (1 - ms_probs[0])
                        else:
                             new_risk = ms_probs[0][dropout_idx]
                             
                        # 3. Display Results
                        delta = new_risk - dropout_prob
                        
                        st.markdown("#### Simulation Results")
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            st.metric("New Risk Score", f"{new_risk:.1%}", f"{delta:.1%}", delta_color="inverse")
                            
                        with res_col2:
                            if new_risk < st.session_state.low_risk_threshold:
                                st.success("Outcome: **SAFE**")
                            elif new_risk > st.session_state.high_risk_threshold:
                                st.error("Outcome: **HIGH RISK**")
                            else:
                                st.warning("Outcome: **MONITOR**")
                                
                    except Exception as e:
                        st.error(f"Simulation failed: {e}")

            
        st.subheader("Model Explainability (SHAP)")
        with st.spinner("Calculating SHAP values..."):
            # Pass the dictionary artifact to calculate_shap_values so it can handle the logic
            # Use a sample of X for background
            explainer, shap_vals, X_transformed, feature_names = calculate_shap_values(
                model_artifact, 
                X.sample(min(100, len(X)), random_state=42) if 'X' in locals() else None, 
                selected_student_data
            )
            
            # SHAP values for the specific class (Dropout)
            if isinstance(shap_vals, list):
                # Multi-class output from KernelExplainer
                if len(shap_vals) > dropout_idx:
                    sv = shap_vals[dropout_idx]
                else:
                    sv = shap_vals[0] # Fallback
            else:
                # Binary / single array
                # For PyGAM binary, this usually explains P(y=1) (e.g., Graduate)
                # If 'Dropout' is Class 0, we need to INVERT the SHAP values 
                # to explain P(Dropout).
                sv = shap_vals
                if probs.ndim == 1 and classes[1] != "Dropout":
                     # Model explains "Graduate". We want "Dropout".
                     # Invert the impact.
                     sv = -1 * sv
                
            # Force Plot / Bar Plot logic
            st.markdown("**Why did the model make this prediction?**")
            
            # Ensure sv is 1D array of feature impacts
            impact_values = np.array(sv)
            if impact_values.ndim > 1:
                impact_values = impact_values.flatten()
            
            # Check lengths
            # For GAMs/OneHot, feature_names might be huge.
            if len(feature_names) != len(impact_values):
                # Re-align - this can happen if variable inputs
                # Try to use just the top ones or handle mismatched shapes gracefully
                st.warning(f"Shape mismatch (feats={len(feature_names)}, impact={len(impact_values)}). Showing raw impacts.")
                feature_names = [f"Feature {i}" for i in range(len(impact_values))]

            # Retrieve the input values for this student
            # X_transformed is returned by calculate_shap_values
            # Ideally X_transformed corresponds to the X_test we passed (student_data)
            # which is 1 row.
            
            student_inputs = X_transformed[0] if X_transformed.ndim > 1 else X_transformed
            
            # Filter Logic:
            # We want to HIDE features that are:
            # 1. Categorical (Course, App Mode)
            # 2. AND have a value of 0 (False)
            
            filtered_data = []
            
            for i, name in enumerate(feature_names):
                clean_name = clean_feature_name(name)
                val = student_inputs[i] if i < len(student_inputs) else 0
                impact = impact_values[i]
                
                # Check criteria
                is_categorical_feature = ("Course" in clean_name or "Application mode" in clean_name or "Tuition" in clean_name)
                is_inactive = (abs(val) < 0.01) # effectively 0
                has_no_impact = (abs(impact) < 0.001) # effectively 0 impact

                if (is_categorical_feature and is_inactive) or has_no_impact:
                    continue # SKIP IT
                
                filtered_data.append({"Feature": clean_name, "Impact": impact})
            
            # Create a DataFrame for plotting
            shap_df = pd.DataFrame(filtered_data)
            
            if not shap_df.empty:
                # Split into Risk (Positive) and Protective (Negative)
                risk_df = shap_df[shap_df["Impact"] > 0].sort_values("Impact", ascending=True)
                protective_df = shap_df[shap_df["Impact"] < 0].sort_values("Impact", ascending=False)
                
                # Unified Scale calculation
                max_val = 0
                if not shap_df.empty:
                    max_val = shap_df["Impact"].abs().max()
                limit = max_val * 1.1 if max_val > 0 else 0.1
                
                # Create Columns - RISK ON LEFT, PROTECTIVE ON RIGHT (Original Layout)
                col_risk, col_prot = st.columns(2)
                
                # Function to calculate consistent height PER PLOT
                def get_plot_height(n_bars):
                    return max(n_bars * 0.5 + 1.0, 2.0)

                # FIXED MARGIN for symmetry
                # We force the left margin to be 40% of the figure width to accommodate labels
                # This ensures the actual plot area starts at the exact same horizontal pixel in both columns
                margin_left = 0.4 

                with col_risk:
                    st.markdown("#### ⚠️ Risk Factors")
                    if not risk_df.empty:
                        h_risk = get_plot_height(len(risk_df))
                        fig_r, ax_r = plt.subplots(figsize=(5, h_risk))
                        
                        # Adjust margin!
                        fig_r.subplots_adjust(left=margin_left, right=0.95, top=0.9, bottom=0.1)
                        
                        ax_r.barh(risk_df['Feature'], risk_df['Impact'], color='red', height=0.6)
                        ax_r.set_xlim([0, limit])
                        ax_r.set_xlabel("Impact (Increases Risk)")
                        
                        # Spines - Clean
                        ax_r.spines['right'].set_visible(False)
                        ax_r.spines['top'].set_visible(False)
                        
                        st.pyplot(fig_r, use_container_width=True)
                    else:
                        st.info("No major risk factors found.")
                        
                with col_prot:
                    st.markdown("#### 🛡️ Protective Factors")
                    if not protective_df.empty:
                        h_prot = get_plot_height(len(protective_df))
                        fig_p, ax_p = plt.subplots(figsize=(5, h_prot))
                        
                        # Adjust margin matches Risk plot
                        fig_p.subplots_adjust(left=margin_left, right=0.95, top=0.9, bottom=0.1)
                        
                        # Protective - Original logic (Bars grow Left -> Right)
                        # Original code used xlim([0, -limit]) which effectively inverts the values visually 
                        # if values are negative.
                        # Wait, protective_df['Impact'] contains negative values.
                        # If we plot negative values on xlim[0, -limit], 0 is left, -limit is right.
                        # So -0.5 is at x=-0.5.
                        # If axis is 0..-1, -0.5 is halfway.
                        # So it grows Left->Right. Correct.
                        
                        ax_p.barh(protective_df['Feature'], protective_df['Impact'], color='green', height=0.6)
                        ax_p.set_xlim([0, -limit]) 
                        ax_p.set_xlabel("Impact (Reduces Risk)")
                        
                        # Spines - Clean (Original had Left/Right hidden? Reverting to standard clean)
                        ax_p.spines['left'].set_visible(True) # Need left spine for tick labels
                        ax_p.spines['right'].set_visible(False)
                        ax_p.spines['top'].set_visible(False)
                        
                        # Restore default tick params (Left side)
                        ax_p.yaxis.tick_left()
                        
                        st.pyplot(fig_p, use_container_width=True)
                    else:
                        st.info("No major protective factors found.")
            else:
                st.info("No significant features influenced this prediction.")
            
    with col2:
        st.subheader("🤖 GenAI Insight")
        
        # Prepare data for GenAI
        top_features = list(zip(shap_df['Feature'], shap_df['Impact'])) if not shap_df.empty else []
        
        explanation = generate_genai_explanation(selected_student_index, dropout_prob, top_features)
        
        st.info(explanation)
        
        st.markdown("### Counselor Actions")
        action = st.selectbox("Recommended Action", ["None", "Send Email", "Schedule Meeting", "Refer to Tutor"])
        
        if st.button("Confirm Action"):
            st.success(f"Action '{action}' recorded for Student {selected_student_index}.")


else:
    st.warning("Please ensure the model is trained and data is available.")
    
if __name__ == "__main__":
    print("WARNING: You are running this script directly with Python.")
    print("Please run this app using: streamlit run predictive_model/dashboard2.py")