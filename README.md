# NeuroVision-Proactive-Stroke-Monitoring

This project implements a machine learning model to predict the risk of brain stroke based on a dataset of patient features. The model uses a combination of **Deep Neural Network (DNN)** and **XGBoost** with a stacking ensemble approach for better prediction accuracy.

## Requirements

- **Python 3.6+**
- **Google Colab** (Recommended)
- Python Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - imbalanced-learn
  - matplotlib
  - seaborn
  - tensorflow
  - imbalanced-learn (for SMOTE)

You can install all required libraries using the following command:
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn tensorflow


## Dataset

The dataset used in this project is `Brain_Stroke_Dataset.csv`, which contains information about several health indicators of patients along with whether they have suffered a stroke or not.

### Features:

- **gender**: Gender of the patient
- **ever_married**: Whether the patient has ever been married
- **work_type**: Type of employment
- **Residence_type**: Type of residence (urban or rural)
- **smoking_status**: Smoking habits of the patient
- **Alcohol Intake**: Information about alcohol consumption
- **Family History of Stroke**: Whether the patient has a family history of stroke
- **bmi**: Body Mass Index
- **HDL** and **LDL**: Cholesterol levels extracted from the `Cholesterol Levels` column

### Target:

- **stroke**: Indicates whether the patient has had a stroke (1) or not (0).

## How to Run the Code

### Step 1: Clone the GitHub Repository
First, clone the repository to your local machine or open it in Colab directly:

git clone https://github.com/your-username/brain-stroke-prediction.git


Alternatively, you can open the `.ipynb` file in **Google Colab**.

### Step 2: Upload the Dataset to Colab

After opening the notebook in Colab:

1. Locate the **Brain_Stroke_Dataset.csv** file from your cloned repository.
2. In Colab, upload the dataset by clicking on the folder icon on the left side to open the file explorer.
3. Click the upload button and select the `Brain_Stroke_Dataset.csv` file.
4. Make sure the file is correctly uploaded, and the path matches the location where you call `pd.read_csv()` in the code.

### Step 3: Execute Cells

Once the dataset is uploaded:

1. Ensure the dataset path in the notebook matches the file location in Colab.
2. Execute each code cell sequentially by clicking **Runtime** > **Run All** or by running cells one by one.

The key steps involved in the notebook are:

- **Data Preprocessing**: Imputation of missing values, encoding categorical variables, normalization, and feature scaling.
- **Model Training**: Training a **Deep Neural Network (DNN)** and **XGBoost** model on the preprocessed data.
- **Stacking**: Combining DNN and XGBoost outputs with a Logistic Regression meta-model.
- **Evaluation**: Evaluating model performance using accuracy, confusion matrix, and ROC curves.

### Step 4: View Results

Once the cells are executed, you will be able to see:

- **Confusion Matrix**: A heatmap of the confusion matrix to understand the model's predictions.
- **ROC Curve**: Receiver Operating Characteristic (ROC) curve to evaluate the classification performance.
- **Accuracy**: Printed accuracy score of the stacked model.
- **Classification Report**: Precision, recall, and F1-score for each class.

## Model Details

- **Deep Neural Network (DNN)**: 
  - Layers: 3 Dense layers with ReLU activation.
  - Dropout: 20% dropout for regularization.
  - Output: Sigmoid activation for binary classification.
  
- **XGBoost**:
  - n_estimators: 200
  - max_depth: 10
  - learning_rate: 0.05
  - subsample: 0.8

- **Stacking**: The DNN and XGBoost outputs are stacked, and a **Logistic Regression** model is used as a meta-learner.

## Results

The stacked model provides predictions based on both DNN and XGBoost outputs and achieves better accuracy compared to individual models.

