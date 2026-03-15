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

from explainability import calculate_shap_values, generate_genai_explanation, generate_simulation_explanation
from llm_helper import get_chat_response

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

# --- Global Action Map ---
action_map = {
    "High Risk": [
        "Create Academic Recovery Plan",
        "Assign Mandatory Tutoring Sessions",
        "Send Academic Improvement Resources",
        "Schedule Meeting: At-risk Student Intervention Meeting",
        "Notify Tutors About High Risk Students"
    ],
    "Medium Risk": [
        "Create Academic Recovery Plan",
        "Send Early Warning Notification to Student",
        "Send Academic Improvement Resources",
        "Schedule Meeting: Advice of Attendance to Lectures & Tutorials"
    ],
    "Likely Graduates": [ # Low Risk
        "Send Academic Improvement Resources",
        "Send Graduation Requirements Checklist",
        "Career Planning Session"
    ],
    "Dean's List Students": [
        "Recommend Mentorship Roles for High Risk Students",
        "Send E-Mail: Dean's List Acceptance"
    ]
}

# Title
# --- Session State Init ---
if "guide_shown" not in st.session_state:
    st.session_state.guide_shown = False
if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = False

@st.dialog("Welcome to Student Success Dashboard", width="large")
def render_welcome():
    """Renders a brief welcome popup explaining the dashboard's purpose and benefits."""
    st.markdown("""
    ### Supporting Counselors in Identifying and Assisting At-Risk Students
    
    This dashboard helps you identify at-risk students early and take proactive interventions to improve student success.
    
    #### What This Dashboard Does:
    
    **Risk Prediction**: Predicts which students are at risk of dropping out based on academic performance, enrollment data, and other key factors.
    
    **Prioritized Student Lists**: Automatically sorts and categorizes students by risk level (High/Medium/Low) so you can focus your efforts where they matter most. Customized sorting is also enabled.
    
    **Tracking & Notes**: Mark students you've tracked and keep intervention notes organized in one place.
    
    **Group Interventions**: Target entire groups of at-risk students with corresponding support strategies.
    
    **Explainable AI**: Shows you exactly why each student is flagged as at-risk, with clear visualizations of contributing factors.
    
    **What-If Scenarios**: You can play with variables and observe how the risk score changes.
    
    ---
    
    Click the ❓ button in the top-right corner anytime for a detailed guide on how to use each feature.
    """)

@st.dialog("Dashboard Guide", width="large")
def render_guide():
    """Renders the detailed dashboard guide."""
    st.markdown("""
    ### Welcome to the Student Success Dashboard!
    
    This tool is designed to help you identify students at risk of dropout early and take proactive measures. 
    Here is a quick overview of how to navigate and use the features:

    #### 1. Data & Configuration (Sidebar)
    *   **Risk Thresholds**: Adjust the `Low` and `High` risk sliders. 
        *   Students with risk **above** the red threshold are flagged as **High Risk**.
        *   Students **below** the green threshold are **Low Risk**.
        *   Those in between are marked as **Medium Risk**.
    *   **AI Recommendations**: Click "AI Threshold Recommendation" to let the system suggest optimal risk cutoffs based on recent validation data.

    #### 2. Filters & List View
    *   **Filter Students**: Use the "Filter Students" dropdown to narrow down the list by *Course*, *Application Mode*, or other attributes like *Tuition Status*.
    *   **Student List**: The table shows students matching your filters.
        *   **Highlights**: Risk Percentages are colored based on risk (Green/Yellow/Red).
        *   **Tracking**: Blue rows indicate students you have already marked/tracked.
        *   **Select**: Click a row to view the detailed profile.

    #### 3. Student Profile & Analysis
    Once a student is selected, you will see:
    *   **Profile Card**: Key academic indicators (Grades, Lectures Enrolled/Passed, Tuition status).
    *   **Risk Status**: A badge indicating if they are High, Medium or Low Risk.
    *   **Tracking/Action**: Mark a student as "Tracked" and leave notes (e.g., "Meeting scheduled").

    #### 4. Explainability (Why this prediction?)
    *   **Risk Score Explanation**: 
        *   **Red Bars (Factors Increasing Dropout Risk)**: Characteristics that *increase* the likelihood of dropout.
        *   **Green Bars (Factors Reducing Dropout Risk)**: Characteristics that *reduce* risk.
    *   **GenAI Insight**: An AI-generated text summary explaining the student's situation in plain language, highlighting key risk factors and protective factors.

    #### 5. GenAI Insight Chatbot
    *   **Interactive Q&A**: Ask follow-up questions about the selected student in natural language.
    *   **Context-Aware Responses**: The chatbot has access to the student's complete profile, including grades, enrollment data, risk factors, and SHAP analysis.
    *   **Examples**: Ask questions like *"Why are their grades low?"*, *"What interventions would help most?"*, or *"How does their age affect dropout risk?"*

    #### 6. What-If Analysis (Ceteris Paribus)
    *   **Simulate Scenarios**: Test hypothetical changes to student attributes such as:
        *   Tuition payment status
        *   Course enrollment
        *   Semester grades
        *   Application mode
        *   Age at enrollment
    *   **Immediate Risk Feedback**: See how the risk score changes with each simulation.
    *   **AI Simulation Analysis**: After running a simulation, receive an AI-generated explanation that:
        *   Interprets *why* the risk changed based on the modified attributes.
        *   Identifies which specific attributes contributed most to the change in risk score.

    ---
    """)

# Auto-Show Welcome on First Load
if not st.session_state.welcome_shown:
    st.session_state.welcome_shown = True
    st.session_state.show_welcome_dialog = True
else:
    if "show_welcome_dialog" not in st.session_state:
        st.session_state.show_welcome_dialog = False

# Initialize guide dialog state
if "show_guide_dialog" not in st.session_state:
    st.session_state.show_guide_dialog = False


# Title
col_title, col_help = st.columns([0.95, 0.05])
with col_title:
    st.title("Student Success & Support Dashboard")
with col_help:
    st.write("") # Spacer
    st.write("")
    # Help Menu
    if st.button("❓", help="Open Dashboard Guide"):
        st.session_state.show_guide_dialog = True
        st.rerun()

# Render dialogs based on state
if st.session_state.show_welcome_dialog:
    st.session_state.show_welcome_dialog = False  # Reset for next run
    render_welcome()
elif st.session_state.show_guide_dialog:
    st.session_state.show_guide_dialog = False  # Reset for next run
    render_guide()

st.markdown("---")

# --- CSS STYLING ---
st.markdown("""
<style>
/* Custom Slider Handle Colors */
/* Left Handle (Safe) - Green */
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"]:nth-of-type(1) {
    background-color: #28a745 !important;
    border-color: #28a745 !important;
    color: #28a745 !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"]:nth-of-type(1) * {
    color: #28a745 !important;
}

/* Right Handle (Risk) - Dark Red */
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"]:nth-of-type(2) {
    background-color: #8b0000 !important;
    border-color: #8b0000 !important;
    color: #8b0000 !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"]:nth-of-type(2) * {
    color: #8b0000 !important;
}

/* Moving Labels */
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"]:nth-of-type(1)::after {
    content: "Low Risk";
    position: absolute;
    top: 20px;
    left: 50%;
    transform: translateX(-50%);
    color: #28a745;
    font-size: 0.75rem;
    font-weight: bold;
    white-space: nowrap;
}

div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"]:nth-of-type(2)::after {
    content: "High Risk";
    position: absolute;
    top: 20px;
    left: 50%;
    transform: translateX(-50%);
    color: #8b0000;
    font-size: 0.75rem;
    font-weight: bold;
    white-space: nowrap;
}

/* RESET for Sliders inside Expanders (Filters) */
div[data-testid="stExpander"] div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
    background-color: #ff4b4b !important; /* Default Streamlit Red */
    border-color: #ff4b4b !important;
    color: inherit !important;
}

div[data-testid="stExpander"] div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] * {
    color: inherit !important;
}

div[data-testid="stExpander"] div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"]::after {
    content: none !important;
}

/* Increase Expander Header Font Size */
div[data-testid="stExpander"] summary p {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<h2 style='font-size: 2rem;'>Configuration</h2>", unsafe_allow_html=True)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# -------------------------------
# AI Threshold Recommendation Modal (Session State)
# -------------------------------


# -------------------------------
# AI Threshold Recommendation Modal (Session State)
# -------------------------------
if "show_threshold_modal" not in st.session_state:
    st.session_state.show_threshold_modal = False

if "high_risk_threshold" not in st.session_state:
    st.session_state.high_risk_threshold = 0.7

if "low_risk_threshold" not in st.session_state:
    st.session_state.low_risk_threshold = 0.3




# Load Data & Model
@st.cache_resource
def load_resources():
    # Paths - Adjusted for GAM model
    # Expecting dashboard.py to be in predictive_model/, so gam/ is a subdir
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gam", "gam_model_student2.joblib")
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")
    
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
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        base = parts[0].replace("_", " ")
        val = parts[1]
        
        # Mapping
        if "Course" in base:
            val = course_map.get(int(val), val)
        elif "Application mode" in base:
             val = application_mode_map.get(int(val), val)
        return f"{base}: {val}"

    # --- SPECIFIC MAPPINGS (Curricular Units) ---
    # Normalize for matching: lower case, remove underscores/spaces
    norm_name = name.lower().replace("_", "").replace(" ", "")
    
    # 1st Semester
    if "curricularunits1stsem(enrolled)" in norm_name:
        return "1st Semester Lectures Enrolled"
    if "curricularunits1stsem(approved)" in norm_name:
        return "1st Semester Lectures Passed"
    if "curricularunits1stsem(grade)" in norm_name:
        return "1st Semester Average Grade"
        
    # 2nd Semester
    if "curricularunits2ndsem(enrolled)" in norm_name:
        return "2nd Semester Lectures Enrolled"
    if "curricularunits2ndsem(approved)" in norm_name:
        return "2nd Semester Lectures Passed"
    if "curricularunits2ndsem(grade)" in norm_name:
        return "2nd Semester Average Grade"

    # Generic Cleanup: Replace underscores with spaces
    return name.replace("_", " ")

model_artifact, df = load_resources()

if model_artifact is not None and df is not None:
    model = model_artifact["model"]
    preprocessor = model_artifact["preprocess"]
    label_encoder = model_artifact["label_encoder"]

    # --- TRACKING DATA HANDLING ---
    tracking_file = os.path.join(os.path.dirname(__file__), "..", "data", "tracking_data.csv")
    
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

    def bulk_update_tracking(indices, notes, is_tracked=1):
        # Load current
        td = load_tracking_data()
        
        # Ensure indices list is unique to avoid potential ambiguities
        indices = list(set(indices))
        
        # Identify missing indices
        # We need to ensure we don't assume index types align perfectly without check,
        # but typically they should be matching if sourced from the same app runtime.
        existing_idx = set(td.index)
        missing_idx = [i for i in indices if i not in existing_idx]
        
        # Append new rows if needed
        if missing_idx:
            new_rows = pd.DataFrame(index=missing_idx, columns=td.columns)
            td = pd.concat([td, new_rows])
        
        # Now safely update
        td.loc[indices, "Is_Tracked"] = is_tracked
        td.loc[indices, "Notes"] = notes
        
        td.index.name = "Student_Index"
        td.to_csv(tracking_file)
        
        # --- SYNC SESSION STATE ---
        # If widgets for these students were already created, update their state
        # so they reflect the new bulk values immediately.
        for idx in indices:
            key_chk = f"trk_chk_{idx}"
            key_note = f"trk_note_{idx}"
            
            if key_chk in st.session_state:
                st.session_state[key_chk] = (is_tracked == 1)
            if key_note in st.session_state:
                st.session_state[key_note] = notes
                
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
    st.sidebar.subheader("Dropout Risk Threshold")
    
    # Threshold Sliders
    risk_range = st.sidebar.slider(
        "Set thresholds for Low & High Risk", 0.0, 1.0, 
        (st.session_state.low_risk_threshold, st.session_state.high_risk_threshold), 
        0.05
    )
    st.session_state.low_risk_threshold, st.session_state.high_risk_threshold = risk_range
    
    
    # -------------------------------
    # AI Threshold Recommendation
    # -------------------------------
    if st.sidebar.button("AI Threshold Recommendation"):
        st.session_state.show_threshold_modal = not st.session_state.show_threshold_modal

    if st.session_state.show_threshold_modal:
        st.sidebar.info("""
        **AI Mockup Analysis**
        
        The predictive engine suggests optimizing thresholds based on recent validation results:
        
        *   **High Risk:** `0.70` -> `0.65`
        *   **Safe:** `0.30` -> `0.35`
        
        *Reasoning: Adjusting sensitivity will capture 12% more at-risk students.*
        """)
        
        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.button("Apply"):
                st.session_state.high_risk_threshold = 0.65
                st.session_state.low_risk_threshold = 0.35
                st.session_state.show_threshold_modal = False
                st.rerun()
        with c2:
            if st.button("Dismiss"):
                st.session_state.show_threshold_modal = False
                st.rerun()
    



    # --- Sidebar Filters ---
    
    # Data Source Selection - Force Existing Student
    data_source = "Select Existing Student"
    
    selected_student_data = None
    selected_student_index = None

    if True: # Existing Student Mode
        
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
                # --- NEW: Unified Filter Section ---
                # User Request: "course and mode inside... name filter students"
                st.sidebar.write("") # Padding
                st.sidebar.write("") # Padding
                with st.sidebar.expander("Filter Students", expanded=False):
                    
                    # 3. Dynamic Attribute Filters
                    # Exclude 'Risk Score' and 'Target' and the ones already filtered above
                    cols_to_exclude = ["Risk Score"]
                    if "target_col" in locals():
                        cols_to_exclude.append(target_col)
                    
                    # Get available columns based on original df logic (but using risk_df for current state)
                    available_attribs = sorted([c for c in risk_df.columns if c not in cols_to_exclude])
                    
                    # Move Course and Apply mode to top if present
                    priority_cols = ["Course", "Application_mode", "Application mode"]
                    sorted_attribs = []
                    for p in priority_cols:
                        if p in available_attribs:
                            sorted_attribs.append(p)
                            available_attribs.remove(p)
                    sorted_attribs.extend(available_attribs)

                    selected_attribs = st.multiselect(
                        "Select Attributes to Filter", 
                        options=sorted_attribs,
                        # Default to Course and App Mode being selected initially? 
                        # User wants them treated as attributes. Typically user expects them valid by default?
                        # Let's keep them unselected by default OR selected by default so logic holds?
                        # Previous logic had them visible by default. 
                        # Let's clean state: default empty? Or default basic ones?
                        # User said "course ve mode attributu de içinde olsun... aralarında bi fark yok"
                        default=[],
                        format_func=lambda x: clean_feature_name(x),
                        key="dynamic_filter_multiselect"
                    )
                    
                    # Initialize filtered DF (Base copy)
                    # We do it here so the loop can modify it
                    filtered_risk_df = risk_df.copy()
                    
                    for attr in selected_attribs:
                        col_data = risk_df[attr]
                        
                        # --- SPECIAL HANDLING FOR COURSE / APP MODE ---
                        if attr == "Course":
                             available_courses = sorted(list(course_map.values()))
                             sel_c = st.selectbox(f"Filter {attr}", ["All"] + available_courses, key="dyn_course")
                             if sel_c != "All":
                                 c_rev = {v: k for k, v in course_map.items()}
                                 c_id = c_rev.get(sel_c)
                                 if c_id:
                                     filtered_risk_df = filtered_risk_df[filtered_risk_df["Course"] == c_id]
                                     # Warning check
                                     curr_count = len(filtered_risk_df)
                                     if 0 < curr_count < 20:
                                         st.warning(f"⚠️ Low sample: {curr_count} students")
                                     
                        elif attr in ["Application_mode", "Application mode"]:
                             available_am = sorted(list(application_mode_map.values()))
                             sel_am = st.selectbox(f"Filter {attr}", ["All"] + available_am, key="dyn_app_mode")
                             if sel_am != "All":
                                 am_rev = {v: k for k, v in application_mode_map.items()}
                                 am_id = am_rev.get(sel_am)
                                 if am_id:
                                     # Use actual column name
                                     col_name = "Application_mode" if "Application_mode" in risk_df.columns else "Application mode"
                                     filtered_risk_df = filtered_risk_df[filtered_risk_df[col_name] == am_id]
                                     # Warning check
                                     curr_count = len(filtered_risk_df)
                                     if 0 < curr_count < 20:
                                         st.warning(f"⚠️ Low sample: {curr_count} students")

                        else:
                            # --- OTHER ATTRIBUTES ---
                            unique_count = col_data.nunique()
                            is_numeric = pd.api.types.is_numeric_dtype(col_data)
                            
                            # Specific check for Tuition or general low cardinality
                            if (unique_count < 10) or (not is_numeric) or (attr == "Tuition_fees_up_to_date"):
                                # Categorical (Multiselect)
                                unique_vals = sorted(col_data.dropna().unique())
                                unique_vals_str = [str(x) for x in unique_vals]
                                
                                sel_vals = st.multiselect(
                                    f"{clean_feature_name(attr)}",
                                    options=unique_vals_str,
                                    default=unique_vals_str,
                                    key=f"dyn_multi_{attr}"
                                )
                                
                                if len(sel_vals) < len(unique_vals_str):
                                    mask = col_data.astype(str).isin(sel_vals)
                                    filtered_risk_df = filtered_risk_df[mask]
                                    
                            else:
                                # Numerical Slider
                                try:
                                    c_num = pd.to_numeric(col_data, errors='coerce')
                                    min_val = float(c_num.min())
                                    max_val = float(c_num.max())
                                    
                                    if pd.isna(min_val) or pd.isna(max_val):
                                         continue

                                    if min_val == max_val:
                                        st.caption(f"{attr}: Constant {min_val}")
                                    else:
                                        rng = st.slider(
                                            f"{clean_feature_name(attr)}", 
                                            min_value=min_val, 
                                            max_value=max_val, 
                                            value=(min_val, max_val),
                                            key=f"slider_{attr}"
                                        )
                                        
                                        filtered_risk_df = filtered_risk_df[
                                            (filtered_risk_df[attr] >= rng[0]) & 
                                            (filtered_risk_df[attr] <= rng[1])
                                        ]
                                except Exception as e:
                                    pass

                st.sidebar.markdown("---")

                # --- NEW: Group Summary ---
                st.subheader("📊 Group Overview")
                
                # Metrics
                total_students = len(filtered_risk_df)
                
                # High Risk
                high_risk_students = filtered_risk_df[filtered_risk_df["Risk Score"] > st.session_state.high_risk_threshold]
                high_risk_count = len(high_risk_students)
                
                # Low Risk (Safe)
                safe_students = filtered_risk_df[filtered_risk_df["Risk Score"] <= st.session_state.low_risk_threshold]
                safe_count = len(safe_students)

                # Medium Risk
                medium_risk_students = filtered_risk_df[
                    (filtered_risk_df["Risk Score"] > st.session_state.low_risk_threshold) & 
                    (filtered_risk_df["Risk Score"] <= st.session_state.high_risk_threshold)
                ]
                medium_risk_count = len(medium_risk_students)
                
                avg_risk = filtered_risk_df["Risk Score"].mean() if total_students > 0 else 0.0
                
                # Disclaimer
                if 0 < total_students < 30:
                    st.warning(f"⚠️ Small Sample Size: This group has only {total_students} students. Statistical insights may be limited.")
                elif total_students == 0:
                    st.warning("No students match the selected filters.")
                    
                # Display Metrics and Intervention in Columns
                gs_col1, gs_col2 = st.columns([2, 1])
                
                with gs_col1:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Total", total_students)
                    m2.metric("High Risk Student Count", high_risk_count, delta_color="inverse")
                    m3.metric("Med. Risk Student Count", medium_risk_count, delta_color="off")
                    m4.metric("Low Risk Student Count", safe_count, delta_color="normal") # Green is good
                    m5.metric("Avg Risk", f"{avg_risk:.1%}")
                
                with gs_col2:
                    # Bulk Intervention
                    with st.expander("Group Intervention"):
                        # Target Selection
                        target_group = st.radio(
                            "Target Group", 
                            ["High Risk", "Medium Risk", "Likely Graduates", "Dean's List Students"], 
                            horizontal=True, 
                            label_visibility="collapsed"
                        )
                        
                        # Unified Action List - Mapped by Target Group
                        # action_map is now global


                        # Get actions for selected group, default to empty if not found
                        common_actions = action_map.get(target_group, [])
                        
                        action_type = st.selectbox("Action", common_actions, key="bulk_action_unified")

                        # Determine target students based on selection
                        target_indices = []
                        target_count = 0
                        
                        if target_group == "High Risk":
                            target_df = high_risk_students
                        elif target_group == "Medium Risk":
                            target_df = medium_risk_students
                        elif target_group == "Dean's List Students":
                            # Top 5% lowest risk
                            # We use filtered_risk_df as source
                             if len(filtered_risk_df) > 0:
                                n_top_5_percent = max(1, int(len(filtered_risk_df) * 0.05))
                                target_df = filtered_risk_df.nsmallest(n_top_5_percent, "Risk Score")
                             else:
                                target_df = pd.DataFrame()
                        else: # Likely Graduates
                            target_df = safe_students
                            
                        target_count = len(target_df)
                            
                        target_count = len(target_df)
                        
                        if target_count > 0:
                            st.write(f"**Target:** {target_count} Students ({target_group})")
                            
                            if st.button(f"Execute Action"):
                                    # SYNC: Update tracking for all these students
                                    target_indices = target_df.index.tolist()
                                    bulk_update_tracking(target_indices, action_type, is_tracked=1)
                                    
                                    if target_group == "Likely Graduates":
                                        st.balloons()
                                    else:
                                        st.success(f"Action executed!")
                                        
                                    st.success(f"'{action_type}' queued for {target_count} students.")
                                    st.rerun()
                            
                            # Reset Button
                            if st.button("Reset Tracking", help="Clear tracking status and notes for these students"):
                                target_indices = target_df.index.tolist()
                                bulk_update_tracking(target_indices, "", is_tracked=0)
                                st.success(f"Tracking reset for {target_count} students.")
                                st.rerun()
                        else:
                            st.info(f"No students found in {target_group}.")
                        
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
                    "Select a Student by Index", 
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
                
                
                # Initialize Session State for columns if needed
                if "risk_list_display_cols" not in st.session_state:
                    st.session_state.risk_list_display_cols = default_cols
                
                display_cols = st.session_state.risk_list_display_cols
                if not display_cols:
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
                # Create a view with friendly names
                display_df = merged_df[display_cols].copy()
                display_df.columns = [clean_feature_name(c) for c in display_df.columns]
                
                st.dataframe(
                    display_df.style
                    .format({"Risk Score": "{:.1%}"})
                    .apply(color_rows, axis=1),
                    height=300,
                    use_container_width=True,
                    key="risk_list_table" 
                )

                # Show/Hide Columns (Moved Below Table)
                with st.expander("Show/Hide Columns"):
                    st.multiselect(
                        "Select Columns to Display", 
                        options=display_candidates,
                        format_func=lambda x: clean_feature_name(x),
                        key="risk_list_display_cols"
                    )
                    if not st.session_state.risk_list_display_cols:
                         st.warning("Please select at least one column.")

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
        st.sidebar.success(f"✅ LOW RISK ({dropout_prob:.1%})")
    else:
        st.sidebar.warning(f"⚠️ MEDIUM RISK ({dropout_prob:.1%})")
            
    # --- Main Content Details ---
    
    st.markdown("---") # Separator between list and details
    
    # --- TRACKING & ACTIONS ---
    # Moved to Student Profile section


    col1, col2 = st.columns([1, 1])
    
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

        # --- NEW CARD LAYOUT (Table Design) ---
        
        # Prepare values
        v_age = int(get_val("Age_at_enrollment")) if get_val("Age_at_enrollment") != "N/A" else "N/A"
        v_t_status = "Up to Date" if get_val("Tuition_fees_up_to_date") == 1 else "Overdue"
        
        def fmt_grade(val_raw):
             return f"{val_raw:.2f}" if val_raw != "N/A" else "N/A"
        
        def fmt_int(val_raw):
             return int(val_raw) if val_raw != "N/A" else "N/A"

        v_g1 = fmt_grade(get_val('Curricular_units_1st_sem_(grade)'))
        v_e1 = fmt_int(get_val('Curricular_units_1st_sem_(enrolled)'))
        v_a1 = fmt_int(get_val('Curricular_units_1st_sem_(approved)'))
        
        v_g2 = fmt_grade(get_val('Curricular_units_2nd_sem_(grade)'))
        v_e2 = fmt_int(get_val('Curricular_units_2nd_sem_(enrolled)'))
        v_a2 = fmt_int(get_val('Curricular_units_2nd_sem_(approved)'))

        # HTML Structure with Borders
        # We use a white background to ensure text visibility
        # Lighter/Thinner border: 2px solid #ccc
        # Removed explicit fonts to inherit default Streamlit styles
        html_card = f"""
        <div style="border: 1px solid rgba(128, 128, 128, 0.5); border-radius: 8px; margin-bottom: 20px; background-color: transparent;">
          <!-- Row 1: Course | App Mode -->
          <div style="display: flex; border-bottom: 1px solid rgba(128, 128, 128, 0.5);">
            <div style="flex: 1; padding: 12px; border-right: 1px solid rgba(128, 128, 128, 0.5);">
               <h4 style="color: #4A90E2; margin: 0;">Course</h4>
               <p style="margin: 5px 0 0 0; color: inherit;">{d_course}</p>
            </div>
            <div style="flex: 1; padding: 12px;">
               <h4 style="color: #4A90E2; margin: 0;">Application Mode</h4>
               <p style="margin: 5px 0 0 0; color: inherit;">{d_app_mode}</p>
            </div>
          </div>

          <!-- Row 2: Age | Tuition -->
          <div style="display: flex; border-bottom: 1px solid rgba(128, 128, 128, 0.5);">
            <div style="flex: 1; padding: 12px; border-right: 1px solid rgba(128, 128, 128, 0.5);">
               <h4 style="color: #4A90E2; margin: 0;">Age at Enrollment</h4>
               <p style="margin: 5px 0 0 0; color: inherit;">{v_age}</p>
            </div>
            <div style="flex: 1; padding: 12px;">
               <h4 style="color: #4A90E2; margin: 0;">Tuition Fees</h4>
               <p style="margin: 5px 0 0 0; color: inherit;">{v_t_status}</p>
            </div>
          </div>

          <!-- Row 3: 1st Sem -->
          <div style="padding: 12px; border-bottom: 1px solid rgba(128, 128, 128, 0.5);">
             <h4 style="color: #E91E63; margin-bottom: 12px; margin-top: 0;">1st Semester Performance</h4>
             <div style="display: flex; justify-content: space-between;">
                <div style="flex: 1;">
                    <h4 style="color: #4A90E2; margin: 0;">Average Grade</h4>
                    <p style="margin: 5px 0 0 0; color: inherit;">{v_g1}</p>
                </div>
                <div style="flex: 1;">
                    <h4 style="color: #4A90E2; margin: 0;">Lectures Enrolled</h4>
                    <p style="margin: 5px 0 0 0; color: inherit;">{v_e1}</p>
                </div>
                <div style="flex: 1;">
                     <h4 style="color: #4A90E2; margin: 0;">Lectures Passed</h4>
                     <p style="margin: 5px 0 0 0; color: inherit;">{v_a1}</p>
                </div>
             </div>
          </div>

          <!-- Row 4: 2nd Sem -->
          <div style="padding: 12px;">
             <h4 style="color: #E91E63; margin-bottom: 12px; margin-top: 0;">2nd Semester Performance</h4>
             <div style="display: flex; justify-content: space-between;">
                <div style="flex: 1;">
                    <h4 style="color: #4A90E2; margin: 0;">Average Grade</h4>
                    <p style="margin: 5px 0 0 0; color: inherit;">{v_g2}</p>
                </div>
                <div style="flex: 1;">
                    <h4 style="color: #4A90E2; margin: 0;">Lectures Enrolled</h4>
                    <p style="margin: 5px 0 0 0; color: inherit;">{v_e2}</p>
                </div>
                <div style="flex: 1;">
                     <h4 style="color: #4A90E2; margin: 0;">Lectures Passed</h4>
                     <p style="margin: 5px 0 0 0; color: inherit;">{v_a2}</p>
                </div>
             </div>
          </div>
        </div>
        """
        
        st.markdown(html_card, unsafe_allow_html=True)
        
        # --- TRACKING & ACTIONS (Moved) ---
        if data_source == "Select Existing Student":
             # Check current status
             # Note: We are already in col1 context
             curr_tracked_val = 0
             curr_notes = ""
             
             if selected_student_index in tracking_df.index:
                 curr_tracked_val = int(tracking_df.loc[selected_student_index, "Is_Tracked"])
                 curr_notes = str(tracking_df.loc[selected_student_index, "Notes"])
                 if curr_notes == "nan": curr_notes = ""
             
             curr_tracked = (curr_tracked_val == 1)

             # Dynamic keys based on student index - Force new unique keys
             key_track = f"trk_chk_{selected_student_index}"
             key_notes = f"trk_note_{selected_student_index}"

             # Callback to save immediately
             def on_track_change():
                 # Read from state using dynamic keys
                 val_c = st.session_state.get(key_track, False)
                 val_n = st.session_state.get(key_notes, "")
                 
                 save_tracking_data(
                     selected_student_index, 
                     val_c, 
                     val_n
                 )
                 
             # UI Components
             st.markdown("#### Action/Tracking")
             t_c1, t_c2 = st.columns([1, 3])
             with t_c1:
                 is_tracked_input = st.checkbox(
                     f"Mark as Tracked (ID: {selected_student_index})", 
                     value=curr_tracked,
                     key=key_track,
                     on_change=on_track_change
                 )
                 if is_tracked_input:
                     st.caption("💾 Saved")
             
             with t_c2:
                 notes_input = st.text_area(
                     "Action Notes", 
                     value=curr_notes,
                     placeholder="Add notes here...",
                     key=key_notes,
                     on_change=on_track_change
                 )
                 
                 # Reset Button
                 def reset_single_student_tracking():
                     # Update session state
                     if key_track in st.session_state: st.session_state[key_track] = False
                     if key_notes in st.session_state: st.session_state[key_notes] = ""
                     # Update persistent data
                     save_tracking_data(selected_student_index, False, "")
                 
                 st.button("Reset", key=f"reset_trk_{selected_student_index}", help="Clear tracking and notes", on_click=reset_single_student_tracking)

        st.markdown("---")
        # Ceteris Paribus moved below SHAP

            
        st.subheader("Risk Score Explanation")
        
        # --- CACHED SHAP CALCULATION ---
        # Cache key based on student ID to prevent re-running on chat interactions
        shap_cache_key = f"shap_cache_{selected_student_index}"
        
        if shap_cache_key not in st.session_state:
            with st.spinner("Calculating SHAP values..."):
                # Pass the dictionary artifact to calculate_shap_values so it can handle the logic
                # Use a sample of X for background
                st.session_state[shap_cache_key] = calculate_shap_values(
                    model_artifact, 
                    X.sample(min(100, len(X)), random_state=42) if 'X' in locals() else None, 
                    selected_student_data
                )
        
        # Load from cache
        explainer, shap_vals, X_transformed, feature_names = st.session_state[shap_cache_key]
            
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
                
                import textwrap
                
                # Helper to wrap long labels
                def wrap_labels(labels, width=50):
                    return [textwrap.fill(label, width) for label in labels]

                # Unified Scale calculation
                max_val = 0
                if not shap_df.empty:
                    max_val = shap_df["Impact"].abs().max()
                limit = max_val * 1.1 if max_val > 0 else 0.1
                
                # Function to calculate consistent height PER PLOT
                def get_plot_height(n_bars):
                    # Slightly taller to accommodate wrapped text
                    return max(n_bars * 0.7 + 1.5, 3.5) # Increased base height to ensure room for margins

                # CRITICAL: Fixed Layout Parameters
                # Using bbox_inches=None requires precise margin control
                FIXED_LEFT_MARGIN = 0.45 
                MARGIN_BOTTOM_INCHES = 1.0 # Fixed space for x-axis label
                MARGIN_TOP_INCHES = 0.3    # Minimal top space
                
                # 1. RISK FACTORS
                st.markdown("#### Factors Increasing Dropout Risk")
                if not risk_df.empty:
                    h_risk = get_plot_height(len(risk_df))
                    # Increased figsize width from 5 to 12
                    fig_r, ax_r = plt.subplots(figsize=(12, h_risk))
                    
                    # Calculate dynamic fractions
                    bottom_frac = MARGIN_BOTTOM_INCHES / h_risk
                    top_frac = 1.0 - (MARGIN_TOP_INCHES / h_risk)
                    
                    # Force exact same margins for both
                    fig_r.subplots_adjust(left=FIXED_LEFT_MARGIN, right=0.95, top=top_frac, bottom=bottom_frac)
                    
                    # Wrap labels
                    r_labels = wrap_labels(risk_df['Feature'])
                    
                    ax_r.barh(r_labels, risk_df['Impact'], color='#FF4B4B', height=0.6) # Streamlit Red
                    ax_r.set_xlim([0, limit])
                    ax_r.set_xlabel("Impact (Increases Risk)")
                    
                    # Spines - Clean
                    ax_r.spines['right'].set_visible(False)
                    ax_r.spines['top'].set_visible(False)
                    ax_r.spines['left'].set_visible(True)
                    ax_r.spines['left'].set_color('#333333')
                    
                    # Increase tick label size for "Bigger" feel
                    ax_r.tick_params(axis='both', which='major', labelsize=11)
                    ax_r.xaxis.label.set_size(12)
                    
                    # CRITICAL: bbox_inches=None prevents re-cropping that ruins alignment
                    st.pyplot(fig_r, use_container_width=True, bbox_inches=None)
                else:
                    st.info("No major Factors Increasing Dropout Risk found.")
                    
                st.write("") # Spacer
                st.write("") 
                        
                # 2. PROTECTIVE FACTORS (Stacked below)
                st.markdown("#### Factors Reducing Dropout Risk")
                if not protective_df.empty:
                    h_prot = get_plot_height(len(protective_df))
                    # Increased figsize width from 5 to 12
                    fig_p, ax_p = plt.subplots(figsize=(12, h_prot))
                    
                    # Calculate dynamic fractions
                    bottom_frac = MARGIN_BOTTOM_INCHES / h_prot
                    top_frac = 1.0 - (MARGIN_TOP_INCHES / h_prot)
                    
                    # Force exact same margins for both
                    fig_p.subplots_adjust(left=FIXED_LEFT_MARGIN, right=0.95, top=top_frac, bottom=bottom_frac)
                    
                    # Wrap labels
                    p_labels = wrap_labels(protective_df['Feature'])
                    
                    ax_p.barh(p_labels, protective_df['Impact'], color='#2E7D32', height=0.6) # Dark Green
                    ax_p.set_xlim([0, -limit]) 
                    ax_p.set_xlabel("Impact (Reduces Risk)")
                    
                    # Spines - Clean
                    ax_p.spines['left'].set_visible(True) 
                    ax_p.spines['left'].set_color('#333333')
                    ax_p.spines['right'].set_visible(False)
                    ax_p.spines['top'].set_visible(False)
                    ax_p.yaxis.tick_left()
                    
                    # Increase tick label size
                    ax_p.tick_params(axis='both', which='major', labelsize=11)
                    ax_p.xaxis.label.set_size(12)
                    
                    # CRITICAL: bbox_inches=None prevents re-cropping that ruins alignment
                    st.pyplot(fig_p, use_container_width=True, bbox_inches=None)
                else:
                    st.info("No major Factors Reducing Dropout Risk found.")
            else:
                st.info("No significant features influenced this prediction.")
        
        st.markdown("---")

        # --- Ceteris Paribus / What-If Analysis (Moved) ---
        with st.expander("What-If Analysis"):
            # Auto-reset form when student changes
            if "last_cp_student" not in st.session_state:
                st.session_state.last_cp_student = None
            
            if st.session_state.last_cp_student != selected_student_index:
                # Student changed - clear form state
                keys = ["cp_tuition", "cp_app_mode", "cp_course", "cp_grade1", "cp_grade2", "cp_app1", "cp_app2", "cp_age"]
                for k in keys:
                    if k in st.session_state:
                        del st.session_state[k]
                st.session_state.last_cp_student = selected_student_index
                st.rerun() # Force rerun to ensure widgets pick up new values
            
            # Wrapper for Reset Logic
            def reset_cp_state():
                keys = ["cp_tuition", "cp_app_mode", "cp_course", "cp_grade1", "cp_grade2", "cp_app1", "cp_app2", "cp_age"]
                for k in keys:
                    if k in st.session_state:
                        del st.session_state[k]

            col_sim_header, col_sim_reset = st.columns([0.85, 0.15])
            with col_sim_header:
                st.markdown("Simulate changes to this student's profile to see how the risk changes.")
            with col_sim_reset:
                st.button("Reset", on_click=reset_cp_state, help="Reset all fields to original values")
            
            # Form for simulation
            with st.form("what_if_form"):
                cp_col1, cp_col2, cp_col3 = st.columns(3)
                
                with cp_col1:
                    # Binary / Categorical features
                    cp_tuition = st.checkbox("Tuition fees up to date", value=(get_val("Tuition_fees_up_to_date") == 1), key="cp_tuition")
                    
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
                        
                    cp_app_mode_label = st.selectbox("Application Mode", options=app_mode_options, index=app_mode_index, key="cp_app_mode")

                    # Course
                    cp_current_course = get_val("Course")
                    if str(cp_current_course).replace('.','').isdigit():
                         cp_current_course_label = course_map.get(int(cp_current_course), "Unknown")
                    else:
                         cp_current_course_label = cp_current_course

                    course_options = list(course_map.values())
                    try:
                         course_index = course_options.index(cp_current_course_label)
                    except:
                         course_index = 0
                    
                    cp_course_label = st.selectbox("Course", options=course_options, index=course_index, key="cp_course")

                with cp_col2:
                    # Numeric - Grades
                    cp_grade1 = st.number_input("1st Semester Average Grade", min_value=0.0, max_value=20.0, value=float(get_val('Curricular_units_1st_sem_(grade)') if get_val('Curricular_units_1st_sem_(grade)') != "N/A" else 0.0), key="cp_grade1")
                    cp_grade2 = st.number_input("2nd Semester Average Grade", min_value=0.0, max_value=20.0, value=float(get_val('Curricular_units_2nd_sem_(grade)') if get_val('Curricular_units_2nd_sem_(grade)') != "N/A" else 0.0), key="cp_grade2")
                    
                    # Age
                    cp_age = st.number_input("Age at enrollment", min_value=17, max_value=70, value=int(get_val('Age_at_enrollment') if get_val('Age_at_enrollment') != "N/A" else 20), key="cp_age")

                with cp_col3:
                     # Numeric - Units Approved
                    cp_app1 = st.number_input("1st Semester Lectures Passed", min_value=0, max_value=30, value=int(get_val('Curricular_units_1st_sem_(approved)') if get_val('Curricular_units_1st_sem_(approved)') != "N/A" else 0), key="cp_app1")
                    cp_app2 = st.number_input("2nd Semester Lectures Passed", min_value=0, max_value=30, value=int(get_val('Curricular_units_2nd_sem_(approved)') if get_val('Curricular_units_2nd_sem_(approved)') != "N/A" else 0), key="cp_app2")
                
                submitted = st.form_submit_button("Simulate New Risk")
                
                if submitted:
                    # 1. Create modified dataframe
                    modified_student = selected_student_data.copy()
                    
                    # Update values
                    modified_student["Tuition_fees_up_to_date"] = 1 if cp_tuition else 0
                    
                    # Map back App Mode label to ID
                    app_mode_rev_map = {v: k for k, v in application_mode_map.items()}
                    modified_student["Application_mode"] = app_mode_rev_map.get(cp_app_mode_label, 1) # Default to 1 if fail
                    
                    # Map back Course label to ID
                    course_rev_map = {v: k for k, v in course_map.items()}
                    modified_student["Course"] = course_rev_map.get(cp_course_label, 1)

                    modified_student["Age_at_enrollment"] = cp_age
                    
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
                                st.success("Outcome: **LOW RISK**")
                            elif new_risk > st.session_state.high_risk_threshold:
                                st.error("Outcome: **HIGH RISK**")
                            else:
                                st.warning("Outcome: **MEDIUM RISK**")
                        
                        # --- GENAI EXPLANATION ---
                        st.markdown("---")
                        
                        # Detect Changes
                        changes_made = {}
                        
                        # Check Tuition
                        old_tuition = get_val("Tuition_fees_up_to_date")
                        new_tuition = 1 if cp_tuition else 0
                        if old_tuition != "N/A" and int(old_tuition) != new_tuition:
                            changes_made["Tuition"] = (old_tuition, new_tuition)
                            
                        # Check Grades
                        old_g1 = get_val("Curricular_units_1st_sem_(grade)")
                        if old_g1 != "N/A" and abs(float(old_g1) - cp_grade1) > 0.1:
                            changes_made["1st Sem Grade"] = (old_g1, cp_grade1)
                            
                        old_g2 = get_val("Curricular_units_2nd_sem_(grade)")
                        if old_g2 != "N/A" and abs(float(old_g2) - cp_grade2) > 0.1:
                            changes_made["2nd Sem Grade"] = (old_g2, cp_grade2)
                            
                        # Check Course/App Mode
                        if modified_student["Course"].iloc[0] != selected_student_data["Course"].iloc[0]:
                            old_course_label = course_map.get(int(selected_student_data["Course"].iloc[0]), selected_student_data["Course"].iloc[0])
                            new_course_label = course_map.get(int(modified_student["Course"].iloc[0]), modified_student["Course"].iloc[0])
                            changes_made["Course"] = (old_course_label, new_course_label)
                        
                        # Check Application Mode
                        if modified_student["Application_mode"].iloc[0] != selected_student_data["Application_mode"].iloc[0]:
                            old_app_label = application_mode_map.get(int(selected_student_data["Application_mode"].iloc[0]), selected_student_data["Application_mode"].iloc[0])
                            new_app_label = application_mode_map.get(int(modified_student["Application_mode"].iloc[0]), modified_student["Application_mode"].iloc[0])
                            changes_made["Application Mode"] = (old_app_label, new_app_label)
                            
                        # Check Age
                        old_age = get_val("Age_at_enrollment")
                        if old_age != "N/A" and int(old_age) != cp_age:
                            changes_made["Age"] = (old_age, cp_age)

                        # Generate Explanation
                        sim_explanation = generate_simulation_explanation(dropout_prob, new_risk, changes_made)
                        st.info(sim_explanation)
                                
                    except Exception as e:
                        st.error(f"Simulation failed: {e}")

    with col2:
        st.subheader("GenAI Insight")
        
        # Prepare data for GenAI
        top_features = list(zip(shap_df['Feature'], shap_df['Impact'])) if not shap_df.empty else []
        
        # 1. Initial Analysis
        explanation = generate_genai_explanation(selected_student_index, dropout_prob, top_features)
        st.info(explanation)
        st.caption("*Disclaimer: AI-generated insights may be inaccurate. Verify with professional judgment.*")
        
        # 2. Interactive Chat
        st.divider()
        st.subheader("AI Chatbot")
        
        # Unique key for this student's chat
        chat_key = f"chat_{selected_student_index}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []
            
        # Display container for chat (scrollable)
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state[chat_key]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat Input
        # Chat Input logic with Rerun Pattern
        if prompt := st.chat_input("Ask about this student...", key=f"input_{selected_student_index}"):
            # 1. Save User Message immediately
            st.session_state[chat_key].append({"role": "user", "content": prompt})
            # 2. Flag to generate response on next run
            st.session_state[f"gen_resp_{selected_student_index}"] = True
            st.rerun()

        # Check if we need to generate a response (Triggered by previous rerun)
        if st.session_state.get(f"gen_resp_{selected_student_index}", False):
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        # Build FULL Context with actual student data
                        risk_level_text = "High Risk" if dropout_prob > st.session_state.high_risk_threshold else "Medium Risk" if dropout_prob > st.session_state.low_risk_threshold else "Low Risk"
                        
                        context_str = f"""=== STUDENT PROFILE ===
Student ID: {selected_student_index}
Risk Level: {risk_level_text} ({dropout_prob:.1%})

Course: {d_course}
Application Mode: {d_app_mode}
Age at Enrollment: {v_age}
Tuition Status: {v_t_status}

1st Semester:
- Average Grade: {v_g1}/20
- Lectures Enrolled: {v_e1}
- Lectures Passed: {v_a1}

2nd Semester:
- Average Grade: {v_g2}/20
- Lectures Enrolled: {v_e2}
- Lectures Passed: {v_a2}

=== KEY RISK/PROTECTIVE FACTORS (SHAP) ===
"""
                        for f, v in top_features[:8]:
                            direction = "↑ increases risk" if v > 0 else "↓ decreases risk"
                            context_str += f"- {f}: {v:.3f} ({direction})\n"
                        
                        messages = [
                            {"role": "system", "content": f"""You are an expert academic counselor assistant for the Student Success Dashboard.

You have access to the following student data:
{context_str}

IMPORTANT:
- Answer questions using the SPECIFIC data provided above.
- If asked about grades, use the actual grade values shown.
- If asked about risk factors, reference the SHAP factors.
- Be professional, empathetic, and concise.
- If the data doesn't contain information to answer a question, say so."""},
                            *st.session_state[chat_key]
                        ]
                        
                        response = get_chat_response(messages)
                        st.markdown(response)
            
            # Save Assistant Response
            st.session_state[chat_key].append({"role": "assistant", "content": response})
            # Clear flag and rerun to finalize state
            st.session_state[f"gen_resp_{selected_student_index}"] = False
            st.rerun()
        

        



else:
    st.warning("Please ensure the model is trained and data is available.")
    
if __name__ == "__main__":
    print("WARNING: You are running this script directly with Python.")
    print("Please run this app using: streamlit run predictive_model/dashboard2.py")
