# Student Success Dashboard - Complete Guide

## Overview
The Student Success Dashboard is a predictive analytics tool designed for university counselors to identify students at risk of dropout early and take proactive interventions.

## Key Terms

### Risk Score
- **Definition**: Probability (0-100%) of student dropout
- **Interpretation**: Higher percentage = higher risk of dropout
- **Usage**: Used to categorize students into risk levels

### Risk Categories
- **High Risk**: Above high threshold (default 70%)
  - Immediate intervention needed
  - Students flagged for priority attention
  
- **Medium Risk**: Between low and high thresholds
  - Monitor closely
  - Preventive measures recommended
  
- **Low Risk**: Below low threshold (default 30%)
  - Student is on track
  - Standard monitoring sufficient

### SHAP Values
- **Definition**: SHapley Additive exPlanations - shows how much each factor contributed to the risk prediction
- **Red Bars**: Factors that increase dropout risk
- **Green Bars**: Protective factors that reduce dropout risk
- **Usage**: Helps counselors understand WHY a student is at risk

### What-If Analysis (Ceteris Paribus)
- **Definition**: Simulation tool to see how changing variables affects risk
- **Purpose**: Test hypothetical scenarios
- **Examples**:
  - "What if this student pays their tuition?"
  - "What if their grades improve?"
  - "What if they enroll in more courses?"

## Dashboard Features

### 1. Configuration (Sidebar)

#### Risk Thresholds
- **Low Risk Threshold**: Adjustable slider (default 30%)
  - Students below this are considered low risk
  - Green indicator
  
- **High Risk Threshold**: Adjustable slider (default 70%)
  - Students above this are considered high risk
  - Red indicator

#### AI Threshold Recommendation
- Click button to get AI-suggested optimal thresholds
- Based on recent validation data
- Helps optimize sensitivity vs. specificity

### 2. Filter Students

Located in sidebar expander, allows filtering by:

- **Course**: 17 different courses including:
  - Nursing, Management, Engineering, Social Service, etc.
  
- **Application Mode**: 18 different admission types including:
  - 1st phase general contingent
  - International student
  - Over 23 years old
  - Transfer students
  - And more

- **Tuition Status**: Whether fees are up to date
  - 1 = Paid
  - 0 = Not paid

- **Other Attributes**: Any numeric or categorical student attribute
  - Age at enrollment
  - Semester grades
  - Lectures enrolled/passed
  - And more

### 3. Group Overview

Displays statistics for the currently filtered group:

- **Total Students**: Count of students matching filters
- **High Risk Count**: Number of students above high threshold
- **Medium Risk Count**: Students between thresholds
- **Low Risk Count**: Students below low threshold
- **Average Risk**: Mean risk score for the group
- **Sample Size Warning**: Alerts if filtered group is too small (<30 students)

### 4. Student List (Risk Overview Table)

Interactive table showing:

- **Student Index**: Unique identifier
- **Risk Score**: Percentage (0-100%)
- **Risk Category**: High/Medium/Low badge
- **Key Attributes**: Course, grades, tuition status, etc.

**Visual Indicators**:
- **Color Coding**: Risk percentages colored (Green/Yellow/Red)
- **Blue Rows**: Students already tracked by counselor
- **Clickable**: Select row to view detailed profile

### 5. Student Profile & Analysis

Once a student is selected:

#### Profile Card
Key academic indicators:
- **Demographics**: Age, course, application mode
- **Academic Performance**:
  - 1st Semester: Enrolled, Passed, Average Grade
  - 2nd Semester: Enrolled, Passed, Average Grade
- **Financial**: Tuition fees up to date status
- **Risk Status**: Badge showing High/Medium/Low

#### Tracking & Actions
- **Mark as Tracked**: Checkbox to flag student
- **Notes Field**: Add intervention notes
  - Examples: "Meeting scheduled", "Sent email", "Referred to counseling"
- **Save**: Persist tracking data to CSV file

### 6. Explainability (Why This Prediction?)

#### Risk Score Explanation Chart
- **SHAP Bar Chart**: Visual breakdown of contributing factors
  - **Red Bars (Right)**: Characteristics increasing dropout risk
    - Example: Low grades, unpaid tuition, few lectures passed
  - **Green Bars (Left)**: Characteristics reducing dropout risk
    - Example: High grades, paid tuition, many lectures passed
  - **Length**: Indicates strength of contribution

#### GenAI Insight
- **AI-Generated Summary**: Plain language explanation
- **Content**:
  - Why the student is at risk
  - Key contributing factors
  - Suggested interventions
- **Powered by**: OpenAI GPT model

### 7. What-If Simulation

Test hypothetical scenarios:

#### Modifiable Attributes
- **Course**: Change to different program
- **Age at Enrollment**: Adjust age
- **Tuition Status**: Toggle paid/unpaid
- **1st Semester**:
  - Lectures Enrolled
  - Lectures Passed
  - Average Grade
- **2nd Semester**:
  - Lectures Enrolled
  - Lectures Passed
  - Average Grade

#### Simulation Results
- **New Risk Score**: Updated prediction
- **Risk Change**: Increase/decrease indicator
- **AI Explanation**: Why the risk changed
  - Realism assessment
  - Real-life interventions that could cause this change

### 8. Group Intervention (Bulk Actions)

Target multiple at-risk students:

- **Select Students**: Choose from filtered list
- **Bulk Notes**: Add same note to all selected
- **Bulk Tracking**: Mark multiple students as tracked
- **Use Case**: Implement group interventions
  - Example: "All high-risk nursing students invited to study group"

### 9. Help System

Two help options (top-right corner):

#### ❓ Dashboard Guide
- Static, comprehensive guide
- Covers all features
- Best for learning the system

#### 🤖 AI Assistant
- Interactive chatbot
- Ask specific questions
- Context-aware help
- Examples:
  - "What does SHAP mean?"
  - "How do I filter by course?"
  - "What's a good intervention for high-risk students?"

## Data Model

### Student Attributes

The dashboard uses the following key attributes:

1. **Application mode** (1-18): Type of admission
2. **Course** (1-17): Academic program
3. **Tuition fees up to date** (0/1): Payment status
4. **Age at enrollment**: Student age when enrolled
5. **Curricular units 1st sem (enrolled)**: Courses enrolled in semester 1
6. **Curricular units 1st sem (approved)**: Courses passed in semester 1
7. **Curricular units 1st sem (grade)**: Average grade semester 1 (0-20 scale)
8. **Curricular units 2nd sem (enrolled)**: Courses enrolled in semester 2
9. **Curricular units 2nd sem (approved)**: Courses passed in semester 2
10. **Curricular units 2nd sem (grade)**: Average grade semester 2 (0-20 scale)

### Target Variable
- **Output**: Student outcome
  - **Dropout**: Student left without completing
  - **Graduate**: Student completed program
  - **Enrolled**: Student still active

## Technical Details

### Model
- **Type**: Generalized Additive Model (GAM)
- **File**: `gam_model_student2_real5fold.joblib`
- **Validation**: 5-fold cross-validation
- **Preprocessing**: Automated pipeline with imputation and scaling

### Data
- **Source**: `student_data_2.csv`
- **Format**: Semicolon-separated (;), comma decimal (,)
- **Size**: 4,425 students
- **Encoding**: UTF-8 with BOM

### Tracking
- **File**: `tracking_data.csv`
- **Persistence**: Local file storage
- **Fields**: Student_Index, Is_Tracked, Notes

## Limitations & Ethical Use

### Limitations
1. **Probabilistic**: Predictions are probabilities, not certainties
2. **Correlation ≠ Causation**: Model identifies patterns, not causes
3. **Historical Data**: Based on past students, may not capture new trends
4. **Professional Judgment**: Should complement, not replace, counselor expertise

### Ethical Guidelines
1. **Support, Don't Stigmatize**: Use predictions to help students, not label them
2. **Equitable Interventions**: Ensure support is fair and accessible to all
3. **Privacy**: Protect student data confidentiality
4. **Transparency**: Be open with students about support programs
5. **Human-Centered**: Keep counselor judgment central to decisions

## Common Workflows

### Workflow 1: Daily Risk Review
1. Open dashboard
2. Filter by "High Risk" students
3. Review new high-risk students (not yet tracked)
4. For each student:
   - View profile and SHAP explanation
   - Read GenAI insight
   - Decide on intervention
   - Mark as tracked with notes

### Workflow 2: Course-Specific Intervention
1. Filter by specific Course
2. Review Group Overview statistics
3. Identify high-risk students in that course
4. Use Group Intervention to:
   - Select all high-risk students
   - Add bulk note: "Invited to [Course] study group"
   - Mark as tracked

### Workflow 3: What-If Planning
1. Select a medium-risk student
2. Open What-If Simulation
3. Test scenarios:
   - Improve grades by 2 points
   - Increase lectures passed
   - Update tuition status
4. Review AI explanation of changes
5. Plan realistic interventions based on simulation
6. Add notes with intervention plan

### Workflow 4: Threshold Optimization
1. Click "AI Threshold Recommendation"
2. Review suggested thresholds
3. Apply if appropriate
4. Monitor impact on student categorization
5. Adjust based on counselor capacity and resources

## Troubleshooting

### Model Load Failure
- **Symptom**: Error message about model file
- **Auto-Fix**: Dashboard attempts to retrain automatically
- **Manual Fix**: Run `gam_real_5fold.py` script

### Missing Data
- **Symptom**: NaN values or extreme numbers
- **Handling**: Automatic preprocessing cleans data
- **Grade Issues**: Values >20 automatically scaled down

### API Key Issues
- **Symptom**: "OpenAI API Key not found" message
- **Fix**: Set `OPENAI_API_KEY` in:
  - `.env` file, OR
  - `.streamlit/secrets.toml` file

### Performance Issues
- **Large Filters**: Narrow down student selection
- **Slow SHAP**: Expected for complex models
- **Browser Memory**: Refresh page if sluggish

## Version Information

- **Dashboard Version**: v2 - 5Fold
- **Model**: GAM (Generalized Additive Model)
- **Framework**: Streamlit
- **Python**: 3.8+
- **Key Dependencies**: pandas, numpy, joblib, shap, matplotlib, openai

## Contact & Support

For technical issues or feature requests, consult:
1. This guide
2. AI Help Assistant (🤖 button)
3. Dashboard Guide (❓ button)
4. System administrator or technical team
