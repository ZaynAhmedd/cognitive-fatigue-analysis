

This project is a data-driven research project that
investigates the relationship between digital device usage patterns and mental performance. I collected the survey data from GoogleForms collected from participants covering variables such as daily screen time, content type consumed, sleep
duration, task-switching frequency, break habits, age group, and physical activity among many other survey questions.


 
I created a Python-based pipeline then cleans the data, performs EDA, and trains multiple machine learning models namely Random Forest, Decision Tree, Gradient Boosting, Linear Regression as well as Ridge Regression to generate visualizatons based on the collected data, as well a synthetic dataset designed on the basis of collected data.


#structure

cognitive-fatigue-analysis/
│
├── clean_and_analyse.py          #main-pipeline
├── generate_dataset.py           #synthetic-dataset-foraccuracy
├── requirements.txt              #contains py dependencies
├── README.md                    
│
├── data/
│   └── fatigue-data.csv          #dataset
│
├── outputs/                      #visualizations
│   ├── screen_time_vs_focus.png
│   ├── correlation_heatmap.png
│   ├── fatigue_distribution.png
│   ├── fatigue_boxplot.png
│   ├── sleep_vs_fatigue.png
│   ├── task_switching_vs_distraction.png
│   ├── content_type_focus.png
│   ├── age_group_analysis.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   └── report.txt
│
├── models/   #all trained ML models
│   └── fatigue_model.pkl        
│
├── notebooks/
│   └── exploratory_analysis.py  
│
├── utils/
│   └── helpers.py               
│
└── tests/
    └── test_pipeline.py    #pytest

-
#quick start

#installing dependencies

bash
pip install -r requirements.txt

#or installing dependencies individually:
bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn pytest
```
#generating the dataset (if fatigue-data.csv is small)

bash
python generate_dataset.py


#this creates `data/fatigue-data.csv` with around 1,200 synthetic survey responses across 15 variables, simulating a real Google Forms export.

#run analysis pipeline

bash
python clean_and_analyse.py


This single command executes the the following complete pipeline:

| Step | Description |
|------|-------------|
| 1. Load & Clean | Loads CSV, handles missing values, clips outliers |
| 2. Feature Engineering | Creates 6 derived features (digital_load_index, sleep_deficit, etc.) |
| 3. EDA | Prints correlation matrices and descriptive statistics |
| 4. Visualisations | Generates 10 publication-quality charts to `outputs/` |
| 5. ML Training | Trains 5 models, evaluates with MAE / RMSE / R² / 5-fold CV |
| 6. Classification | Classifies participants as Low / Moderate / High fatigue risk |
| 7. Prediction | Runs a sample fatigue risk prediction with the best model |

#run supplementary analysis (optional)

bash
python notebooks/exploratory_analysis.py

#Run unit tests (optional)

bash
python -m pytest tests/ -v

#Output files

| File | Description |
|------|-------------|
| `outputs/screen_time_vs_focus.png` | Scatter + grouped bar: screen time vs cognitive outcomes |
| `outputs/correlation_heatmap.png` | Full Pearson correlation matrix heatmap |
| `outputs/fatigue_distribution.png` | Histograms + KDE for fatigue, focus, distraction |
| `outputs/fatigue_boxplot.png` | Boxplots: fatigue vs content type, break frequency, activity, age |
| `outputs/sleep_vs_fatigue.png` | Scatter + bar: sleep duration vs fatigue & focus |
| `outputs/task_switching_vs_distraction.png` | Scatter + bar: task switching analysis |
| `outputs/content_type_focus.png` | Horizontal bar + grouped: focus by content type |
| `outputs/age_group_analysis.png` | Screen time and cognitive metrics by age group |
| `outputs/feature_importance.png` | ML feature importance ranking |
| `outputs/model_comparison.png` | Side-by-side MAE / RMSE / R² across all 5 models |
| `models/fatigue_model.pkl` | Serialised best ML model (pickle) |



#dataset variables

| Variable | Type | Description |
|----------|------|-------------|
| `screen_time` | Float | Daily screen time in hours (0.5–14) |
| `content_type` | Categorical | Primary content consumed |
| `sleep_hours` | Float | Average nightly sleep (3–10 hrs) |
| `task_switches` | Integer | Task switches per hour (0–30) |
| `takes_breaks` | Binary | Yes / No |
| `focus_rating` | Integer | Self-rated focus (1–10) |
| `fatigue_level` | Integer | Self-rated fatigue (1–10) |
| `distraction_score` | Integer | Self-rated distraction (1–10) |
| `age_group` | Categorical | 16-18 / 18-22 / 22-25 / 25-30 / 30+ |
| `primary_device` | Categorical | Smartphone / Laptop / Tablet / Desktop / Multiple |
| `work_study_hours` | Float | Daily work/study hours |
| `social_media_platforms` | Integer | Number of platforms actively used (1–6) |
| `break_frequency` | Categorical | Never / Rarely / Sometimes / Frequently / Always |
| `caffeine_intake` | Integer | Caffeinated drinks per day (0–5) |
| `physical_activity` | Categorical | Sedentary / Light / Moderate / Active / High |

#ml models

five regression models are trained to predict `fatigue_risk_score`:

1.random forest regressor — 200 trees, max_depth=10
2. decision tree regressor — max_depth=8
3. gradient boosting regressotr — 150 estimators, lr=0.08
4.linear regression — OLS baseline
5.ridge regression — L2 regularisation (α=1.0)

evaluation metrics: **MAE, RMSE, R², 5-fold Cross-Validation R²**

the best-performing model is automatically selected and saved to `models/fatigue_model.pkl`.

#these are the findings
-screen time ↔ focus: Strong negative correlation—higher screen time which leads directly to way lower focus

- sleep ↔ fatigue: Strong negative correlation— more sleep correlated to less fatigue and higher efficienvy

-task switching ↔ Distraction:Moderate positive correlation

-content type: Study Content consumers score highest in focus, whereas Short-form Reels score the lowest in focus related metrics.

-break Frequency: Participants who never take breaks report significantly higher fatigue and stress as well.

-physical activity: Sedentary participants show markedly higher fatigue levels, compared to active participants that have drastically reduced fatigue levels

coded and developed and authored by-
zain ahmed
