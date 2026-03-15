# Dashboard Documentation Context

## Overview
The Student Success Dashboard is a predictive analytics tool that helps university counselors identify students at risk of dropping out. It uses a Generalized Additive Model (GAM) trained on historical student data.

## Key Terminology

### Risk Score
A probability (0-100%) representing the likelihood that a student will drop out. Higher = more risk.

### Risk Categories
- **High Risk**: Above the high threshold (default 70%). Immediate intervention recommended.
- **Medium Risk**: Between thresholds. Proactive check-in advised.
- **Low Risk**: Below low threshold (default 30%). No immediate action needed.

### SHAP Values
SHAP (SHapley Additive exPlanations) shows HOW MUCH each factor contributed to the risk score. 
- Positive impact (red bars) = increases dropout risk
- Negative impact (green bars) = decreases dropout risk (protective)

### What-If / Ceteris Paribus
Simulation tool to answer "What if X changed?". Modify student attributes and see the predicted new risk.

## Dashboard Features

### Sidebar
- **Risk Thresholds**: Sliders to define Low/High risk cutoffs.
- **AI Threshold Recommendation**: Suggests optimal thresholds based on model validation.
- **Filter Students**: Filter by Course, Application Mode, Tuition Status, etc.

### Group Overview
Summary statistics for the filtered group: total students, counts per risk category, average risk.

### Student List
Table of students sorted by risk. Colored by risk level. Blue rows = tracked students.

### Student Profile
Detailed view of a selected student: demographics, academic performance, tuition status.

### Risk Score Explanation
Two bar charts:
1. Factors INCREASING dropout risk (red)
2. Factors DECREASING dropout risk (green)

### GenAI Insight
AI-generated summary of the student's situation and suggested next steps.

### Tracking
Mark students as "Tracked" and add notes (e.g., "Scheduled meeting for Jan 15").

### Simulation (What-If)
Change values (grades, tuition, etc.) and see how the risk score would change.

## Model Limitations
- Predictions are probabilistic, not deterministic.
- The model was trained on historical data; it may not perfectly predict future outcomes.
- Correlation is not causation: a factor influencing risk doesn't mean changing it will definitely change the outcome.
- Always combine model insights with professional judgment.

## Ethical Use
- Use this tool to SUPPORT students, not to label or stigmatize them.
- Risk scores should inform outreach, not punish students.
- Ensure interventions are equitable across demographics.
