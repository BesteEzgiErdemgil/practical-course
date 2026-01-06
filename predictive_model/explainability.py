import shap
import pandas as pd
import numpy as np

def calculate_shap_values(model_artifact, X_train, X_test):
    """
    Calculates SHAP values for the test set, handling both Pipeline and GAM dictionary.
    
    Args:
        model_artifact: The trained pipeline or GAM dictionary artifact.
        X_train: Training data (needed for background distribution).
        X_test: Test data to explain.
        
    Returns:
        explainer: The SHAP explainer object.
        shap_values: The calculated SHAP values.
        X_test_transformed: Transformed test data.
        feature_names: Names of features after transformation.
    """
    
    # Check if it's our GAM dictionary artifact
    if isinstance(model_artifact, dict) and "model" in model_artifact and "preprocess" in model_artifact:
        model = model_artifact["model"]
        preprocessor = model_artifact["preprocess"]
        # label_encoder = model_artifact["label_encoder"] # Not needed for SHAP calculation directly
        
        # Transform data
        # Ensure we cast to float for pygam
        X_train_transformed = preprocessor.transform(X_train).astype(float)
        X_test_transformed = preprocessor.transform(X_test).astype(float)
        
        # Get feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
             # Fallback if get_feature_names_out isn't available or fails
             feature_names = [f"feat_{i}" for i in range(X_train_transformed.shape[1])]

        # GAMs are additive, but pygam's implementation with sklearn API usually requires KernelExplainer
        # for general probability outputs.
        # Summarize background to speed up (GAMs can be slow with KernelExplainer)
        # Using 10-20 k-means centroids as background is standard for speed/approximation
        background = shap.kmeans(X_train_transformed, 10) 
        
        # We explain the probability of the positive class (usually index 1)
        # gam.predict_proba returns (N, 2) or (N,). We need a function that returns the prob.
        f = lambda x: model.predict_proba(x)
        
        explainer = shap.KernelExplainer(f, background)
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test_transformed)
        
        return explainer, shap_values, X_test_transformed, feature_names

    # Legacy support for sklearn Pipeline (Baseline Model)
    elif hasattr(model_artifact, 'named_steps'):
        model = model_artifact
        classifier = model.named_steps['clf']
        preprocessor = model.named_steps['pre']
        
        # Transform data
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Get feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            feature_names = None

        if feature_names is None or len(feature_names) != X_train_transformed.shape[1]:
             feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]

        if "Forest" in str(type(classifier)) or "Tree" in str(type(classifier)):
             explainer = shap.TreeExplainer(classifier)
             shap_values = explainer.shap_values(X_test_transformed)
        else:
            background = shap.kmeans(X_train_transformed, 10) 
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
            shap_values = explainer.shap_values(X_test_transformed)
            
        return explainer, shap_values, X_test_transformed, feature_names

    else:
        raise ValueError("Model format not recognized. Expected GAM dictionary or sklearn Pipeline.")

def generate_genai_explanation(student_id, risk_prob, top_features):
    """
    Generates a natural language explanation for a student's risk.
    
    Args:
        student_id: ID of the student.
        risk_prob: Probability of dropout (0.0 to 1.0).
        top_features: List of tuples (feature_name, shap_value) representing key drivers.
        
    Returns:
        str: A natural language summary.
    """
    # MOCK GENAI RESPONSE
    # In a real scenario, this would call OpenAI/Gemini API
    
    risk_level = "High" if risk_prob > 0.6 else "Medium" if risk_prob > 0.3 else "Low"
    
    explanation = f"""**Student Analysis (ID: {student_id})**
    
**Risk Level:** {risk_level} ({risk_prob:.1%})

**Key Drivers:**
"""
    
    for feature, impact in top_features[:3]:
        direction = "increases risk" if impact > 0 else "decreases risk"
        explanation += f"- **{feature}**: This factor {direction} (Impact: {impact:.2f}).\n"
        
    explanation += "\n**GenAI Summary:**\n"
    explanation += f"Based on the predictive model, this student shows a {risk_level.lower()} likelihood of dropping out. "
    
    if risk_level == "High":
        explanation += "The primary concerns are related to their academic performance and financial factors. " \
                       "Immediate intervention is recommended to discuss support options."
    elif risk_level == "Medium":
        explanation += "While not critical, there are some warning signs. A check-in email would be beneficial " \
                       "to ensure they are on the right track."
    else:
        explanation += "The student appears to be progressing well. No immediate action is required, " \
                       "but positive reinforcement is always helpful."
                       
    return explanation

def generate_simulation_explanation(old_risk, new_risk, changes):
    """
    Generates a context-aware explanation for a simulated risk change.
    
    Args:
        old_risk: Original dropout probability (float).
        new_risk: Simulated dropout probability (float).
        changes: Dictionary of changed features {FeatureName: (OldVal, NewVal)}.
        
    Returns:
        str: Markdown explanation.
    """
    delta = new_risk - old_risk
    improvement = -delta
    
    explanation = f"### AI Simulation Analysis\n\n"
    
    if abs(delta) < 0.01:
        explanation += "The simulated changes had **minimal impact** on the risk score. This suggests that the factors modified are not the primary drivers of dropout risk for this student, or the change magnitude was insufficient."
        return explanation

    # Significant change
    direction = "reduced" if improvement > 0 else "increased"
    explanation += f"**Result:** Risk {direction} by **{abs(improvement):.1%}**.\n\n"
    
    explanation += "**Why did the risk change?**\n"
    
    # Analyze specific drivers based on 'changes' dict keys
    # We look for keywords in the changed feature names
    
    change_keys_str = " ".join(changes.keys()).lower()
    
    if "grade" in change_keys_str or "approved" in change_keys_str:
        explanation += "The improvement is largely driven by better **academic performance**. The model weighs recent semester grades heavily as they are strong indicators of engagement and capability.\n\n"
        explanation += "**Intervention Reality Check:**\n"
        explanation += "Achieving this grade improvement typically requires:\n"
        explanation += "*   **Remedial Coursework:** Enrolling in support classes.\n"
        explanation += "*   **Tutor Follow-up:** Regular weekly sessions.\n"
        explanation += "*   **Attendance Enforcement:** Strict monitoring of lecture presence.\n"
        
    elif "tuition" in change_keys_str:
        explanation += "Bringing **tuition fees up to date** removes a significant barrier. Financial instability is a key dropout predictor.\n\n"
        explanation += "**Intervention Reality Check:**\n"
        explanation += "This change implies resolving financial holds. However, **tuition payment alone** often does not solve underlying academic issues. Ensure the student also has academic support.\n"
        
    elif "course" in change_keys_str or "application" in change_keys_str:
        explanation += "Changing the **Course** or **Application Mode** fundamentally shifts the risk profile. Different programs have varying retention rates.\n\n"
        explanation += "**Is this realistic?**\n"
        explanation += "This is a **structural change**, not an intervention. Use this to compare risk across different potential paths for a student (e.g., if they are considering a transfer).\n"
        
    elif "age" in change_keys_str:
         explanation += "The simulation adjusted the **Age at enrollment**.\n\n"
         explanation += "**Note:** Age is immutable. This simulation shows how age demographics correlate with risk, but you cannot 'intervene' to change a student's age.\n"
         
    else:
        explanation += "The model reacted to the combination of adjusted parameters.\n"

    explanation += "\n> _These insights help actionable planning rather than just hypothetical matching._"
    
    return explanation
