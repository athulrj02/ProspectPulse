# ProspectPulse

**ProspectPulse** is a data-driven project that analyzes customer behavior for a bank's marketing campaign using Python. It focuses on predicting whether a customer will subscribe to a term deposit based on various demographic, financial, and marketing features.

---

## Key Features

- **Exploratory Data Analysis (EDA):**
  - Analyzing categorical and numerical features.
  - Visualizations including distribution plots, bar charts, and correlation heatmaps for feature insights.

- **Machine Learning Models:**
  - Implementation of various supervised learning algorithms:
    - Support Vector Machines (SVM)
    - Random Forest Classifier
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Naive Bayes (GaussianNB and BernoulliNB)
    - Decision Trees
    - AdaBoost

- **Model Evaluation:**
  - Comparison of models based on accuracy, precision, recall, F1-score, and confusion matrices.
  - Highlighted **Random Forest** and **Support Vector Machines (SVM)** as top-performing models.

- **Hyperparameter Tuning:**
  - Used **RandomizedSearchCV** to optimize model performance.
  - Tuned hyperparameters for Random Forest and SVM for improved accuracy.

- **Synthetic Minority Oversampling Technique (SMOTE):**
  - Applied SMOTE to handle data imbalance, ensuring fairer model training.

- **Real-Time Prediction:**
  - Used the trained **Random Forest model** to make predictions for new customer data.

---

## Project Workflow

1. **Data Preprocessing:**
   - Cleaned and preprocessed the dataset.
   - Encoded categorical features using **OneHotEncoder**.
   - Standardized numerical features using **StandardScaler**.

2. **Data Splitting:**
   - Split the data into training and testing sets.

3. **Model Implementation:**
   - Trained multiple classifiers and compared their performance.

4. **Hyperparameter Tuning:**
   - Performed cross-validation to find the best hyperparameters.

5. **Prediction:**
   - Deployed the best model (Random Forest) to predict term deposit subscription for new customer data.

---

## Tools and Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - Data Processing: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`
  - Sampling: `imbalanced-learn`

- **Dataset:** [bank.csv](https://github.com/athulrj02/ProspectPulse/blob/master/bank.csv)

---

## Repository Structure

```
ProspectPulse/
│
├── CstData.ipynb          # Jupyter Notebook containing the full code for the project
├── bank.csv               # Dataset used for the analysis
├── Idea.txt               # Project idea and notes
├── Rpaper.docx            # Research paper summarizing the findings
├── README.md              # Project documentation (this file)
```

---

## Results

- **Random Forest Classifier**:
  - Achieved the highest accuracy and balanced performance.
  - Best hyperparameters: 
    - `n_estimators`: 200
    - `min_samples_split`: 2
    - `min_samples_leaf`: 1
    - `max_depth`: 30
    - `criterion`: 'gini'

- **Support Vector Machine**:
  - Competitive accuracy after hyperparameter tuning with the following parameters:
    - `C`: 1
    - `gamma`: 'scale'
    - `kernel`: 'rbf'

- **Prediction Output for New Data:**
  - Predicted output: **Yes** (Customer likely to subscribe to term deposit)

---

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/athulrj02/ProspectPulse.git
   cd ProspectPulse
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook CstData.ipynb
   ```

4. Execute the notebook cells to reproduce the results.

---

## Conclusion

- The Random Forest model, trained on a balanced dataset, demonstrated the highest accuracy and balanced performance across precision, recall, and F1-score.
- The project can serve as a valuable tool for predicting customer behavior, aiding in targeted marketing strategies.
- Continuous monitoring and updates to the model are recommended as new data becomes available.
