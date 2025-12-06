# Lifestyle Data Correlation Analysis and Calorie Burn Prediction

ECE 143 – Programming for Data Analysis (Fall 2025)

## Group Members

- Anthony Henderson (anhenderson@ucsd.edu)  
- Shunkai Yu (shyu@ucsd.edu)  
- Riqian Hu (rih006@ucsd.edu)  
- Yuzhou Ren (yur004@ucsd.edu)  
- Huayang Yu (huy016@ucsd.edu)

---

## Project Overview

This project explores how everyday lifestyle factors – including diet type, workout behavior, and basic demographics – relate to overall health and energy expenditure.

Using a Kaggle Lifestyle dataset (20,000 workout records plus meal metadata), we:

- Perform exploratory data analysis and visualizations
- Analyze correlations between lifestyle variables and health/fitness metrics
- Train a Random Forest classifier to predict calorie burn categories:
  **Low, Medium, High, Very High**
- Interpret feature importance to identify which factors matter most

The final outcome combines visual storytelling and statistical insights to highlight how data-driven analysis can support healthier lifestyle choices.

---

# File Structure 

├── data/
│   ├── Final_data.csv
│   ├── Life_Style_Data_Cleaned.csv
│   └── meal_metadata.csv
│
├── figures/
│   ├── SHAP_contributions_by_gender.png
│   ├── SHAP_summary_bar.png
│   ├── calories_burned_by_workout_type.png
│   ├── corr_calorie_intake_and_expenditure.png
│   ├── corr_matrix_of_key_variables.png
│   ├── diet_vs_rating_violin_with_trend.png
│   ├── key_categorical_distributions.png
│   ├── water_vs_rating_binned.png
│   ├── water_vs_rating_linear.png
│   └── workout_intensity_vs_mood_rating.png
│
├── notebook/
│   └── lifestyle_datamining.ipynb
│
├── Presentation/
│   ├── ECE 143 Final Project.pdf
│   └── draft.md
│
├── src/
│   ├── dataset.py
│   └── preprocess.py
│
└── README.md
