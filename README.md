# Advanced Breast Cancer Diagnosis AI

This is an interactive web application built with Streamlit for real-time breast cancer diagnosis. The app leverages machine learning to predict whether a tumor is benign or malignant based on its features. It also provides tools for exploratory data analysis, model comparison, and explainable AI to understand the predictions.

![App Screenshot](https://placehold.co/800x450/2c2c2c/ffffff?text=Screenshot+of+the+Diagnosis+AI)

---

## ✨ Features

-   **Exploratory Data Analysis (EDA):** Interactive tools to explore the dataset, including feature distributions, descriptive statistics, and a correlation heatmap.
-   **Multi-Model Comparison:** Train and evaluate three different machine learning models:
    -   Logistic Regression
    -   Random Forest
    -   Support Vector Machine (SVM)
-   **Model Performance Dashboard:** View detailed performance metrics for each model, including accuracy, a confusion matrix, and a classification report.
-   **Feature Importance:** For the Random Forest model, visualize the most influential features in determining a diagnosis.
-   **Interactive Diagnosis:** Use intuitive sliders to input patient data and receive an instant diagnosis prediction.
-   **Explainable AI (XAI):** Integrated with SHAP (SHapley Additive exPlanations) to provide a visual explanation for every prediction, showing how each feature contributed to the outcome.
-   **Advanced Visualizations:**
    -   **Principal Component Analysis (PCA):** A 2D plot to visualize the separation of benign and malignant tumors.
    -   **Interactive 3D Scatter Plot:** Explore the relationships between three different features in a 3D space.

---

## 🚀 How to Run

To run this application on your local machine, follow these steps:

### 1. Prerequisites

-   Python 3.7+
-   A Kaggle account and API token. ([How to set up Kaggle API](https://www.kaggle.com/docs/api))

### 2. Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/belajdel/Breast-Cancer-Dignosis-Machine-Learning-Project-](https://github.com/belajdel/Breast-Cancer-Dignosis-Machine-Learning-Project-)
    cd Breast-Cancer-Dignosis-Machine-Learning-Project-
    ```

2.  **Install the required libraries:**
    ```bash
    pip install streamlit pandas scikit-learn seaborn matplotlib plotly shap kagglehub
    ```

3.  **Download the Dataset:**
    Run the following command in your terminal. This will download the dataset from Kaggle and place it in the correct directory.
    ```python
    import kagglehub
    import os

    # Download the latest version of the dataset
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("wasiqaliyasir/breast-cancer-dataset")
    print("Download complete. Path:", path)

    # Move the CSV to the project's root directory for the app to find it
    source_file = os.path.join(path, 'Breast_cancer_dataset.csv')
    destination_file = 'Breast_cancer_dataset.csv'

    if os.path.exists(source_file):
        os.rename(source_file, destination_file)
        print(f"Dataset successfully moved to: {os.path.abspath(destination_file)}")
    else:
        print(f"Error: Could not find the dataset CSV at {source_file}")

    ```

### 3. Running the Application

1.  Once the setup is complete, run the Streamlit app from your terminal:
    ```bash
    streamlit run app.py
    ```

The application will open in your default web browser.

---

## 📊 Dataset

This project uses the **Breast Cancer Wisconsin (Diagnostic) Data Set** from Kaggle. The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

---

## 🛠️ Technologies Used

-   **Streamlit:** For building the interactive web application.
-   **Pandas:** For data manipulation and analysis.
-   **Scikit-learn:** For machine learning (model training, preprocessing, and metrics).
-   **Matplotlib & Seaborn:** For static data visualizations.
-   **Plotly Express:** For interactive 3D visualizations.
-   **SHAP:** For model explainability and generating prediction insights.
-   **KaggleHub:** For programmatic dataset access.
