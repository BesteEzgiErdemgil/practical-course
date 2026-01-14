import shap
from llm_helper import get_chat_response
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
    Generates a natural language explanation for a student's risk using LLM.
    
    Args:
        student_id: ID of the student.
        risk_prob: Probability of dropout (0.0 to 1.0).
        top_features: List of tuples (feature_name, shap_value) representing key drivers.
        
    Returns:
        str: A natural language summary.
    """
    
    # Real GenAI Implementation
    risk_level = "High" if risk_prob > 0.6 else "Medium" if risk_prob > 0.3 else "Low"
    
    # Construct a summary string of the top factors
    factors_desc = ""
    for feature, impact in top_features[:5]: # increased context
        direction = "increasing risk" if impact > 0 else "decreasing risk"
        factors_desc += f"- {feature} ({direction}, impact: {impact:.3f})\n"
        
    prompt_messages = [
        {"role": "system", "content": "You are an expert data analyst for student success. Summarize the student's risk profile for a counselor. Be concise and actionable. Limit to 3-4 sentences."},
        {"role": "user", "content": f"Student ID: {student_id}. Risk Level: {risk_level} ({risk_prob:.1%}).\nKey Factors:\n{factors_desc}\n\nPlease explain why this student is at risk and suggest 1 key next step."}
    ]
    
    # Call OpenAI
    response = get_chat_response(prompt_messages)
    
    # Fallback if API fails
    if "API Key" in response or "Error" in response:
         fallback_expl = f"**AI Analysis Unavailable:** {response}\n\n*Using Offline Fallback:*\n"
         fallback_expl += f"**Risk Level:** {risk_level} ({risk_prob:.1%})\n"
         fallback_expl += "**Key Factors:**\n" + factors_desc
         return fallback_expl
         
    return response

def generate_simulation_explanation(old_risk, new_risk, changes):
    """
    Generates a context-aware explanation for a simulated risk change using LLM.
    
    Args:
        old_risk: Original dropout probability (float).
        new_risk: Simulated dropout probability (float).
        changes: Dictionary of changed features {FeatureName: (OldVal, NewVal)}.
        
    Returns:
        str: Markdown explanation.
    """
    delta = new_risk - old_risk
    improvement = -delta
    
    # Minimal change - no need for LLM
    if abs(delta) < 0.01:
        return "### AI Simulation Analysis\n\nThe simulated changes had **minimal impact** on the risk score. This suggests that the factors modified are not the primary drivers of dropout risk for this student, or the change magnitude was insufficient."

    # Build changes description
    changes_desc = ""
    for var, (old_val, new_val) in changes.items():
        changes_desc += f"- {var}: {old_val} → {new_val}\n"
    
    direction = "reduced" if improvement > 0 else "increased"
    
    prompt_messages = [
        {"role": "system", "content": """You are an expert academic data analyst explaining simulation results.

CRITICAL RULES:
1. ONLY discuss the specific variables that were changed.
2. START by identifying the **Primary Driver** (the single changed variable with biggest impact).
3. Explain WHY this variable affects dropout risk (academic correlation).
4. Briefly mention minor drivers if applicable.
5. Do NOT suggest interventions.
6. Keep response under 100 words."""},
        {"role": "user", "content": f"""A student's dropout risk changed from {old_risk:.1%} to {new_risk:.1%} ({direction} by {abs(improvement):.1%}).

VARIABLES CHANGED:
{changes_desc}

Task:
1. Identify the Primary Driver of this change.
2. Explain the correlation between these specific changes and dropout risk."""}
    ]
    
    response = get_chat_response(prompt_messages)
    
    # Fallback if API fails
    if "API Key" in response or "Error" in response:
        # Use basic rule-based fallback
        explanation = f"### AI Simulation Analysis\n\n**Result:** Risk {direction} by **{abs(improvement):.1%}**.\n\n"
        explanation += f"**Variables Changed:**\n{changes_desc}\n"
        explanation += "_AI explanation unavailable. Please check API configuration._"
        return explanation
    
    return f"### AI Simulation Analysis\n\n**Result:** Risk {direction} by **{abs(improvement):.1%}**.\n\n{response}"

