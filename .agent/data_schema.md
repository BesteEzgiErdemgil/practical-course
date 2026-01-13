# Student Data Schema

## Overview
This document describes the structure and content of the student dataset used in the Student Success Dashboard.

## File Information

- **Filename**: `student_data_2.csv`
- **Location**: `predictive_model/gam/student_data_2.csv`
- **Format**: CSV (Semicolon-separated)
- **Decimal Separator**: Comma (,)
- **Encoding**: UTF-8 with BOM
- **Total Records**: 4,425 students
- **Total Columns**: 11 (10 features + 1 target)

## Column Definitions

### 1. Application mode
- **Type**: Categorical (Integer codes 1-18)
- **Description**: Type of admission/application process
- **Values**:
  - `1`: 1st phase - general contingent
  - `2`: Ordinance No. 612/93
  - `3`: 1st phase - special contingent (Azores Island)
  - `4`: Holders of other higher courses
  - `5`: Ordinance No. 854-B/99
  - `6`: International student (bachelor)
  - `7`: 1st phase - special contingent (Madeira Island)
  - `8`: 2nd phase - general contingent
  - `9`: 3rd phase - general contingent
  - `10`: Ordinance No. 533-A/99 (Different Plan)
  - `11`: Ordinance No. 533-A/99 (Other Institution)
  - `12`: Over 23 years old
  - `13`: Transfer
  - `14`: Change of course
  - `15`: Technological specialization diploma holders
  - `16`: Change of institution/course
  - `17`: Short cycle diploma holders
  - `18`: Change of institution/course (International)

### 2. Course
- **Type**: Categorical (Integer codes 1-17)
- **Description**: Academic program/course of study
- **Values**:
  - `1`: Biofuel Production Technologies
  - `2`: Animation and Multimedia Design
  - `3`: Social Service (evening attendance)
  - `4`: Agronomy
  - `5`: Communication Design
  - `6`: Veterinary Nursing
  - `7`: Informatics Engineering
  - `8`: Equinculture
  - `9`: Management
  - `10`: Social Service
  - `11`: Tourism
  - `12`: Nursing
  - `13`: Oral Hygiene
  - `14`: Advertising and Marketing Management
  - `15`: Journalism and Communication
  - `16`: Basic Education
  - `17`: Management (evening attendance)

### 3. Tuition fees up to date
- **Type**: Binary (0/1)
- **Description**: Whether student's tuition payments are current
- **Values**:
  - `1`: Tuition fees are paid/up to date
  - `0`: Tuition fees are not paid/overdue
- **Importance**: Strong predictor of dropout risk

### 4. Age at enrollment
- **Type**: Numeric (Integer)
- **Description**: Student's age when they enrolled in the program
- **Range**: Typically 18-70 years
- **Distribution**: Most students 18-25, some mature students 30+
- **Note**: Mature students (>23) may have different risk profiles

### 5. Curricular units 1st sem (enrolled)
- **Type**: Numeric (Integer)
- **Description**: Number of courses/units the student enrolled in during 1st semester
- **Range**: 0-18 units
- **Typical**: 5-8 units for full-time students
- **Note**: 0 may indicate special programs (e.g., Animation & Multimedia Design)

### 6. Curricular units 1st sem (approved)
- **Type**: Numeric (Integer)
- **Description**: Number of courses/units the student passed in 1st semester
- **Range**: 0 to (enrolled count)
- **Importance**: Key predictor - low pass rate indicates risk
- **Calculation**: approved/enrolled = pass rate

### 7. Curricular units 1st sem (grade)
- **Type**: Numeric (Float)
- **Description**: Average grade for 1st semester courses
- **Scale**: 0-20 (Portuguese grading system)
  - `0-9`: Fail
  - `10-13`: Pass
  - `14-16`: Good
  - `17-20`: Excellent
- **Special Values**:
  - `0.0`: No courses passed or no grade available
- **Data Issues**: Some values stored in scientific notation (e.g., `1,34286E+16`)
  - Dashboard automatically fixes these to 0-20 scale

### 8. Curricular units 2nd sem (enrolled)
- **Type**: Numeric (Integer)
- **Description**: Number of courses/units enrolled in 2nd semester
- **Range**: 0-18 units
- **Note**: May differ from 1st semester due to course progression

### 9. Curricular units 2nd sem (approved)
- **Type**: Numeric (Integer)
- **Description**: Number of courses/units passed in 2nd semester
- **Range**: 0 to (enrolled count)
- **Importance**: Trend analysis (improving vs. declining performance)

### 10. Curricular units 2nd sem (grade)
- **Type**: Numeric (Float)
- **Description**: Average grade for 2nd semester courses
- **Scale**: 0-20 (Portuguese grading system)
- **Special Values**: Same as 1st semester
- **Data Issues**: Same scientific notation issues, auto-corrected

### 11. Output (Target Variable)
- **Type**: Categorical (String)
- **Description**: Student outcome/status
- **Values**:
  - `Dropout`: Student left program without completing
  - `Graduate`: Student successfully completed program
  - `Enrolled`: Student still actively enrolled
- **Distribution** (approximate):
  - Dropout: ~32%
  - Graduate: ~50%
  - Enrolled: ~18%

## Data Quality Notes

### Known Issues

1. **Scientific Notation in Grades**
   - **Problem**: Some grade values stored as `1,34286E+16` instead of `13.4286`
   - **Cause**: Decimal point formatting error in source data
   - **Solution**: Dashboard automatically detects and corrects values >20
   - **Method**: Recursive division by 10 until value ≤20

2. **Comma Decimal Separator**
   - **Problem**: CSV uses comma (,) for decimals instead of period (.)
   - **Example**: `13,5` instead of `13.5`
   - **Solution**: Dashboard reads with `decimal=","` parameter

3. **Missing Values**
   - **Occurrence**: Some students have 0 enrolled courses (special programs)
   - **Handling**: Imputation in preprocessing pipeline
   - **Note**: 0 grades may be legitimate (no courses) or missing data

4. **Extreme Outliers**
   - **Problem**: Values >1e10 due to formatting errors
   - **Solution**: Automatic scaling by dividing by 1e15

### Data Preprocessing Pipeline

The dashboard applies these transformations automatically:

1. **Load**: Read CSV with semicolon separator and comma decimals
2. **Clean Columns**: Remove BOM characters and extra whitespace
3. **Numeric Conversion**: Convert string-numeric columns to proper numeric type
4. **Outlier Correction**: Fix extreme values (>1e10)
5. **Grade Scaling**: Ensure all grades are 0-20 range
6. **Imputation**: Fill missing values using model's preprocessor
7. **Encoding**: One-hot encode categorical variables (Course, Application mode)

## Sample Data

### Example Student Records

#### High Risk Student (Dropout)
```
Application mode: 8 (2nd phase - general contingent)
Course: 2 (Animation and Multimedia Design)
Tuition fees up to date: 1 (Yes)
Age at enrollment: 20
Curricular units 1st sem (enrolled): 0
Curricular units 1st sem (approved): 0
Curricular units 1st sem (grade): 0.0
Curricular units 2nd sem (enrolled): 0
Curricular units 2nd sem (approved): 0
Curricular units 2nd sem (grade): 0.0
Output: Dropout
```

#### Low Risk Student (Graduate)
```
Application mode: 1 (1st phase - general contingent)
Course: 12 (Nursing)
Tuition fees up to date: 1 (Yes)
Age at enrollment: 18
Curricular units 1st sem (enrolled): 7
Curricular units 1st sem (approved): 7
Curricular units 1st sem (grade): 13.75
Curricular units 2nd sem (enrolled): 8
Curricular units 2nd sem (approved): 8
Curricular units 2nd sem (grade): 14.35
Output: Graduate
```

## Feature Importance

Based on SHAP analysis, typical feature importance ranking:

1. **Curricular units 2nd sem (approved)** - Most important
2. **Curricular units 1st sem (approved)** - Very important
3. **Curricular units 2nd sem (grade)** - Very important
4. **Curricular units 1st sem (grade)** - Very important
5. **Tuition fees up to date** - Important
6. **Age at enrollment** - Moderate
7. **Course** - Moderate (varies by program)
8. **Application mode** - Lower importance

## Usage in Dashboard

### Risk Prediction
- All 10 features used as input to GAM model
- Preprocessor handles encoding and scaling
- Output: Probability of dropout (0-1)

### Filtering
- Course and Application mode: Dropdown selection
- Numeric features: Slider range selection
- Tuition status: Multiselect (0/1)

### What-If Analysis
- All features modifiable except Application mode
- Real-time prediction update
- SHAP explanation regenerated

### Display Names

The dashboard uses friendly names for display:

| Technical Name | Display Name |
|---------------|--------------|
| Curricular units 1st sem (enrolled) | 1st Semester Lectures Enrolled |
| Curricular units 1st sem (approved) | 1st Semester Lectures Passed |
| Curricular units 1st sem (grade) | 1st Semester Average Grade |
| Curricular units 2nd sem (enrolled) | 2nd Semester Lectures Enrolled |
| Curricular units 2nd sem (approved) | 2nd Semester Lectures Passed |
| Curricular units 2nd sem (grade) | 2nd Semester Average Grade |
| Tuition fees up to date | Tuition Status |
| Age at enrollment | Age at Enrollment |

## Data Access

### For AI Agents
- **File Path**: `c:\Users\beste\OneDrive\Masaüstü\praktikum\practical-course\predictive_model\gam\student_data_2.csv`
- **Read Method**: 
  ```python
  pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
  ```

### For Users
- Data loaded automatically by dashboard
- No direct file access needed
- Tracking data saved to `tracking_data.csv`

## Privacy & Security

- **Anonymization**: Student IDs are numeric indices, not real identifiers
- **Sensitive Data**: No personally identifiable information (PII)
- **Access Control**: Local file system permissions
- **Backup**: Recommended for tracking data

## Updates & Maintenance

- **Data Refresh**: Update CSV file to include new student records
- **Model Retraining**: Run `gam_real_5fold.py` after data updates
- **Validation**: Check for data quality issues after updates
- **Tracking**: Tracking data persists across dashboard sessions
