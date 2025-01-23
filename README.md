# Report on FIFA Players' Market Value Prediction

Introduction:
This Report focuses on the Analysis of dataset which contains attributes of Football Players. Our goal is to develop a Predictive model for the Market value of Players using the attributes available in the Dataset. Our dataset includes attributes like Overall rating, height, weight, etc. of 17,954 football players.

Data Overview:
Entries & Features: The dataset comprises 17,954 entries and 51 features.
Key Features: Includes Overall Rating, Age, Positions, Skill Ratings, financial information, etc.
Target Variables: My main focus is on Value_Euro, representing each player's Market Value.

Data Cleaning & Preprocessing:
Outlier Analysis: Almost 10,000 instances have been removed as a part of Outlier Analysis.
Handling Missing Values: Missing values in Value_Euro have been imputed using Median values.

Feature Engineering: Created features such as age_to_potential_ratio, attacking_skills, etc.
The New features are a summarized version of the Existing Features. The existing features are combined by assigning weights to them as per their importance.

Exploratory Data Analysis (EDA):
Correlation Analysis: Formulated the Correlation Matrix for all the attributes and plotted Scatter Plots for a few attributes against value_euro.
Diagrammatic Representation: Box Plots and Density Plots have been plotted to find out the univariate distribution for all the numerical variables.
Statistical Summary: Measures of Central Tendency and Dispersion have been described.

Model Development:
Models Used: Polynomial Regression with Ridge Regularisation, Random ForestRegressor, X-Gradient Boosting.
Data Split: The data was split into an 80-10-10 training, test and validation set.

Model Evaluation:
Evaluation Metrics: Used MSE, RMSE, MAE, R², MAPE to evaluate model performance.
Performance Comparison: Compared the performance of each model.

Model Interpretation:
Polynomial Regression & Random Forest Regressor tends to overfit the data. On the other hand, XG Boost Regressor gave consistent results across Train and Test data.

Conclusions:
Key Findings: XG Boost Regressor is the best performing model among the 3.
Limitations: Some factors like Reputation, Brand Value & Form changes are not considered.
Recommendations: Provided actionable recommendations based on findings.
Future Work: Can try to resolve the limitations stated above.
