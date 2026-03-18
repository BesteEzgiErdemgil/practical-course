# Student Success & Support Dashboard
> **From Prediction to Prevention: Empowering Counselors with Explainable AI**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/GenAI-OpenAI-412991.svg)](https://openai.com/)
[![EDA](https://img.shields.io/badge/Analysis-EDA-orange.svg)](#-data-analysis--exploration)

An interactive, human-centered **Streamlit dashboard** designed to help university counselors identify at-risk students and intervene early. By combining advanced predictive modeling with transparency through Explainable AI (XAI) and Generative AI (GenAI), this tool transforms raw student data into actionable insights.

---

## The Team

This project was developed by a team of four dedicated students:

- **Aybike Altunbaş** 
- **Beste Ezgi Erdemgil** 
- **Defne Elagöz** 
- **Utku Yılmaz** 

for the *Practical Course: Enhancing Data Analysis with Generative AI — Focus on Human Systems* at Technical University of Munich.

---

## The Challenge & Solution

### The Challenge
Student dropout is a complex, multi-factor issue often identified too late. Counselors frequently lack real-time, interpretable data to prioritize interventions effectively.

### Our Solution
We built a predictive system that not only flags "who" is at risk but explains **"why"** and suggests **"what to do."** This bridge between data and action is facilitated by:
1.  **Predictive Modeling**: High-accuracy risk scoring using Generalized Additive Models (GAM).
2.  **Explainability**: Local feature importance via SHAP values.
3.  **Generative AI**: Natural language insights and a virtual assistant for deeper analysis.

---

## Key Features

### 1. Smart Risk Scoring & Thresholds
- Robust predictive engine calculating a **Risk Score (0-100%)** for each student.
- Dynamic threshold adjustment (High/Medium/Low risk) with **AI-powered recommendations** to optimize sensitivity.

### 2. Explainable AI (SHAP)
- Transparency for every prediction.
- Visual breakdown of contributing factors (e.g., tuition status, specific grades, application mode) using **SHAP values**.
- No "black box" decisions; counselors see exactly what drove the risk score.

### 3. Generative AI Insights & Assistant
- **GenAI Explanations**: Automated plain-language summaries of a student's risk profile.
- **Context-Aware Chatbot**: An interactive assistant that helps counselors query student data and explore intervention strategies in natural language.

### 4. What-If Analysis (Simulation)
- A "Ceteris Paribus" simulation tool allowing counselors to modify student attributes (e.g., "What if this student pays their tuition?") and observe immediate changes in the risk score.

---

## Data Analysis & Exploration

Before building the predictive model, we conducted extensive Exploratory Data Analysis (EDA) and statistical testing to understand the factors driving student success:

- **Univariate & Categorical Analysis**: Detailed breakdown of student demographics and their correlation with graduation vs. dropout rates.
- **Feature Selection**: Statistical filtering to identify the most impactful predictors (e.g., tuition status, 1st and 2nd semester academic performance).
- **Statistical Distributions**: Comprehensive analysis of data distribution, skewness, and normality to optimize model performance.

*All research notebooks can be found in the `data analysis/` directory.*

---

## Technical Stack

- **Frontend**: Streamlit (Python-based interactive dashboard)
- **Data Processing**: `Pandas`, `NumPy`
- **Machine Learning**: `PyGAM` (Generalized Additive Models), `Scikit-learn`
- **Explainability**: `SHAP` (Shapley Additive Explanations)
- **Generative AI**: `OpenAI GPT-4` (via API)

---

## Ethical AI & Human Oversight

This system is designed to **augment**, not replace, human judgment.
- **Counselor Decision**: The AI provides data-driven recommendations, but the final intervention decision remains with the counselor.
- **Fairness & Guardrails**: The dashboard includes fairness warnings (e.g., highlighting small sample sizes) and emphasizes using scores to *support* rather than penalize students.

---

## Installation & Setup

To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone [your-repo-url]
   cd practical-course
   ```

2. **Initialize Environment**:
   It is recommended to use a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys**:
   Create a `.env` file in the `src` directory (or modify the existing one) and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_actual_key_here
   ```

5. **Run the Dashboard**:
   ```bash
   streamlit run src/final_dashboard.py
   ```

## Visual Documentation

### Dashboard Overview
*High-level view of student risk metrics and group-level interventions.*
<p align="center">
  <img src="https://github.com/user-attachments/assets/4c7729ae-d4a6-4061-984c-197956a0d879" width="800" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/6e40d649-212f-412f-9110-686ef136274d" width="800" />
</p>

### Student Analysis & GenAI Support
*Detailed student profile featuring SHAP-based risk explanations and an interactive AI assistant.*
<p align="center">
  <img src="https://github.com/user-attachments/assets/cc1e6d02-78c7-4d13-9ed5-156170c0f7b2" width="800" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/65d8ff09-7ca9-41d0-b053-0e95daa4b366" width="800" />
</p>

### What-If Simulation & Results
*Dynamic simulation of student scenarios to predict how specific changes impact dropout risk.*
<p align="center">
  <img src="https://github.com/user-attachments/assets/5cf204a3-d4df-4637-876e-26d0d2c5232e" width="800" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/851d50f7-018d-44c9-957e-fc8dd8892221" width="800" />
</p>

---

## Data Source

The project utilizes the [Predict Students' Dropout and Academic Success](https://www.kaggle.com/datasets/marouandaghmoumi/dropout-and-success-student-data-analysis) dataset from Kaggle, which contains comprehensive records of student academic performance, demographics, and socio-economic factors.

---


