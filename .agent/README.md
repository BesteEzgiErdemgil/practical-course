# AI Agent Resources for Student Success Dashboard

## Purpose
This directory contains documentation and resources specifically designed for AI agents (like ChatGPT, Claude, Gemini, etc.) to understand and assist with the Student Success Dashboard project.

## Available Resources

### 1. Dashboard Guide (`dashboard_guide.md`)
**What it contains:**
- Complete overview of all dashboard features
- Detailed explanations of key terms (Risk Score, SHAP, What-If Analysis)
- Step-by-step workflows for common tasks
- Technical details about the model and implementation
- Troubleshooting guide
- Ethical guidelines and limitations

**When to use:**
- Understanding how the dashboard works
- Explaining features to users
- Troubleshooting issues
- Planning new features or modifications
- Writing documentation

### 2. Data Schema (`data_schema.md`)
**What it contains:**
- Complete CSV file structure and format
- All column definitions with value ranges
- Data quality issues and how they're handled
- Sample student records
- Feature importance rankings
- Data preprocessing pipeline details

**When to use:**
- Understanding the student data
- Analyzing data quality issues
- Planning data-related features
- Debugging data loading problems
- Explaining predictions to users

### 3. Sample Data (`sample_students.csv`)
**What it contains:**
- 20 representative student records
- Mix of Dropout, Graduate, and Enrolled outcomes
- Various risk profiles (high, medium, low)
- Different courses and application modes

**When to use:**
- Quick reference without loading full dataset
- Testing scenarios
- Creating examples for documentation
- Understanding data patterns

## How AI Agents Should Use These Resources

### For General Questions
1. **Read** `dashboard_guide.md` for feature explanations
2. **Reference** specific sections based on user questions
3. **Provide** concise answers with context

### For Data Questions
1. **Consult** `data_schema.md` for column definitions
2. **Check** `sample_students.csv` for real examples
3. **Explain** data quality issues if relevant

### For Troubleshooting
1. **Review** troubleshooting section in dashboard guide
2. **Check** data quality notes in schema
3. **Suggest** specific solutions based on documentation

### For Feature Development
1. **Understand** existing features from guide
2. **Review** data structure from schema
3. **Plan** changes that align with current architecture
4. **Consider** ethical guidelines and limitations

## File Locations

### Documentation (This Directory)
```
.agent/
├── README.md (this file)
├── dashboard_guide.md
├── data_schema.md
└── sample_students.csv
```

### Dashboard Code
```
predictive_model/
├── dashboard2.py (main dashboard)
├── explainability.py (SHAP and AI explanations)
├── llm_helper.py (OpenAI integration)
└── gam/
    ├── student_data_2.csv (full dataset - 4,425 students)
    └── gam_model_student2_real5fold.joblib (trained model)
```

### Tracking Data
```
predictive_model/
└── tracking_data.csv (counselor notes and tracked students)
```

## Quick Reference

### Key Concepts
- **Risk Score**: 0-100% probability of dropout (higher = more risk)
- **SHAP**: Explains which factors contribute to risk
- **What-If**: Simulate changes to see impact on risk
- **Tracking**: Mark students and add intervention notes

### Data Columns (10 features + 1 target)
1. Application mode (1-18)
2. Course (1-17)
3. Tuition fees up to date (0/1)
4. Age at enrollment
5. Curricular units 1st sem (enrolled)
6. Curricular units 1st sem (approved)
7. Curricular units 1st sem (grade) [0-20 scale]
8. Curricular units 2nd sem (enrolled)
9. Curricular units 2nd sem (approved)
10. Curricular units 2nd sem (grade) [0-20 scale]
11. **Output**: Dropout / Graduate / Enrolled

### Common Risk Factors
**High Risk Indicators:**
- Low grades (<10)
- Few courses passed
- Unpaid tuition
- 0 enrolled courses

**Protective Factors:**
- High grades (>14)
- All courses passed
- Paid tuition
- Consistent performance across semesters

## AI Agent Best Practices

### 1. Context Awareness
- Always consider which part of the dashboard the user is asking about
- Reference specific features by their correct names
- Use terminology from the guide (e.g., "What-If Analysis" not "simulation")

### 2. Accuracy
- Don't make up features that don't exist
- Verify information against the documentation
- Acknowledge limitations when uncertain

### 3. Clarity
- Explain technical concepts in plain language
- Provide examples from sample data when helpful
- Break down complex workflows into steps

### 4. Ethical Considerations
- Always mention that predictions are probabilistic, not certain
- Emphasize the importance of professional judgment
- Remind users about equitable treatment of students

### 5. Practical Help
- Suggest specific workflows from the guide
- Reference exact button names and locations
- Provide troubleshooting steps when relevant

## Example AI Agent Responses

### Question: "How do I filter students by course?"
**Good Response:**
"To filter students by course:
1. Open the sidebar
2. Expand the 'Filter Students' section
3. In the 'Select Attributes to Filter' dropdown, choose 'Course'
4. A new dropdown will appear with all 17 courses (Nursing, Management, Engineering, etc.)
5. Select the course you want to view

The student list will update automatically to show only students in that course. You'll also see updated Group Overview statistics for that filtered group."

### Question: "What does SHAP mean?"
**Good Response:**
"SHAP stands for SHapley Additive exPlanations. In the dashboard, SHAP values show you exactly which factors are contributing to a student's dropout risk:

- **Red bars (pointing right)**: Factors that increase dropout risk
  - Example: Low grades, unpaid tuition, few courses passed
  
- **Green bars (pointing left)**: Protective factors that reduce risk
  - Example: High grades, paid tuition, many courses passed

The length of each bar shows how much that factor contributes to the prediction. This helps counselors understand WHY a student is flagged as at-risk, not just that they are at risk."

### Question: "Can I see the actual student data?"
**Good Response:**
"The full student dataset is located at:
`predictive_model/gam/student_data_2.csv`

It contains 4,425 student records with 10 features each. For privacy, students are identified by numeric indices, not real names.

I can also show you the sample data in `.agent/sample_students.csv` which has 20 representative students if you want to see examples without loading the full file.

Note: The data is semicolon-separated with comma decimals, so you'll need to read it with:
```python
pd.read_csv(path, sep=';', decimal=',', encoding='utf-8-sig')
```"

## Updates and Maintenance

### When to Update These Docs
- New dashboard features added
- Data schema changes
- New workflows identified
- Bug fixes that affect user experience
- Model updates or changes

### How to Update
1. Edit the relevant markdown file
2. Keep formatting consistent
3. Add examples where helpful
4. Update the version information if applicable

## Version Info
- **Created**: January 2026
- **Dashboard Version**: v2 - 5Fold
- **Last Updated**: 2026-01-13

## Questions?
If you're an AI agent and encounter something not covered in these docs, you can:
1. Check the actual code in `predictive_model/dashboard2.py`
2. Review the help context in the dashboard code (search for `DASHBOARD_HELP_CONTEXT`)
3. Acknowledge the limitation and ask the user for clarification
