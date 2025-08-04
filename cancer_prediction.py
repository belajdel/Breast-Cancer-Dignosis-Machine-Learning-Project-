import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import plotly.express as px
import lime
import lime.lime_tabular
import numpy as np

@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the breast cancer dataset."""
    df = pd.read_csv(file_path)
    if 'Unnamed: 32' in df.columns:
        df = df.drop('Unnamed: 32', axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

@st.cache_resource
def train_model(_df, model_name):
    """Trains a selected model and returns the model, scaler, and LIME explainer."""
    X = _df.drop(['id', 'diagnosis'], axis=1)
    y = _df['diagnosis']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "Support Vector Machine":
        model = SVC(probability=True, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        class_names=['Benign', 'Malignant'],
        mode='classification'
    )

    return model, scaler, accuracy, conf_matrix, class_report, feature_names, explainer

@st.cache_data
def get_model_comparison(_df):
    """Trains all models and returns a dataframe with their performance metrics."""
    X = _df.drop(['id', 'diagnosis'], axis=1)
    y = _df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }

    results = []

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision (Malignant)": report['1']['precision'],
            "Recall (Malignant)": report['1']['recall'],
            "F1-Score (Malignant)": report['1']['f1-score']
        })
    
    return pd.DataFrame(results)


def main():
    st.set_page_config(page_title="Breast Cancer Diagnosis AI", layout="wide", initial_sidebar_state="expanded")
    
    st.sidebar.title("Breast Cancer Diagnosis AI")
    st.sidebar.markdown("An advanced application for diagnosis, model comparison, and explainable AI.")


    try:
        df = load_data("Breast_cancer_dataset.csv")
    except FileNotFoundError:
        st.sidebar.error("The `Breast_cancer_dataset.csv` file was not found. Please make sure it's in the same directory as the script.")
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["📊 Exploratory Data Analysis", "🤖 Model Performance", "🩺 Make a Diagnosis", "🔬 Simulations & Visualizations"])

    if page == "📊 Exploratory Data Analysis":
        st.header("📊 Exploratory Data Analysis (EDA)")
        st.markdown("Explore the dataset to understand its structure, distributions, and correlations.")

        with st.expander("View Raw Data"):
            st.write(df)

        with st.expander("Descriptive Statistics"):
            st.write(df.describe())

        st.subheader("Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            st.info("Diagnosis Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='diagnosis', data=df, ax=ax)
            ax.set_xticklabels(['Benign (0)', 'Malignant (1)'])
            st.pyplot(fig)

        with col2:
            st.info("Feature Distributions by Diagnosis")
            feature = st.selectbox("Select a feature to visualize:", df.drop(['id', 'diagnosis'], axis=1).columns)
            fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
            sns.histplot(data=df, x=feature, hue='diagnosis', kde=True, ax=ax_dist)
            plt.legend(title='Diagnosis', labels=['Malignant', 'Benign'])
            st.pyplot(fig_dist)

        with st.expander("Feature Correlation Heatmap"):
            st.markdown("This heatmap shows how different features are related to each other. A high value (close to 1.0) means a strong positive correlation.")
            fig_corr, ax_corr = plt.subplots(figsize=(18, 14))
            sns.heatmap(df.drop(['id'], axis=1).corr(), annot=True, fmt='.1f', cmap='coolwarm', ax=ax_corr)
            st.pyplot(fig_corr)

    elif page == "🤖 Model Performance":
        st.header("🤖 Machine Learning Model Performance")
        st.markdown("Compare different models and view their detailed performance metrics.")

        st.subheader("Model Comparison")
        with st.spinner("Training and comparing all models..."):
            comparison_df = get_model_comparison(df)
        
        st.dataframe(comparison_df.set_index('Model'))

        fig_comp = px.bar(comparison_df, x='Model', y='Accuracy', color='Model',
                          title='Model Accuracy Comparison', text_auto='.3f')
        fig_comp.update_layout(yaxis_title='Accuracy Score')
        st.plotly_chart(fig_comp, use_container_width=True)

        st.subheader("Detailed Metric Comparison (for Malignant Class)")
        fig_metrics, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        sns.barplot(x='Model', y='Precision (Malignant)', data=comparison_df, ax=axes[0], palette='viridis')
        axes[0].set_title('Precision for Malignant Diagnosis')
        axes[0].set_ylabel('Precision')
        axes[0].tick_params(axis='x', rotation=15)

        sns.barplot(x='Model', y='Recall (Malignant)', data=comparison_df, ax=axes[1], palette='plasma')
        axes[1].set_title('Recall for Malignant Diagnosis')
        axes[1].set_ylabel('Recall')
        axes[1].tick_params(axis='x', rotation=15)

        sns.barplot(x='Model', y='F1-Score (Malignant)', data=comparison_df, ax=axes[2], palette='magma')
        axes[2].set_title('F1-Score for Malignant Diagnosis')
        axes[2].set_ylabel('F1-Score')
        axes[2].tick_params(axis='x', rotation=15)

        plt.tight_layout()
        st.pyplot(fig_metrics)

        st.subheader("Individual Model Analysis")
        st.markdown("Select a model from the dropdown to see its specific performance details.")
        model_name = st.selectbox("Select a Model for Detailed Analysis", ["Random Forest", "Logistic Regression", "Support Vector Machine"])
        
        model, _, accuracy, conf_matrix, class_report, feature_names, _ = train_model(df, model_name)
        
        st.info(f"Showing details for **{model_name}** | **Accuracy:** {accuracy:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)
        with col2:
            st.write("#### Classification Report")
            st.table(pd.DataFrame(class_report).transpose())

        if model_name == "Random Forest":
            with st.expander("View Feature Importance for Random Forest"):
                st.markdown("This chart shows the most influential features for the Random Forest model.")
                importance_df = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
                importance_df = importance_df.sort_values('importance', ascending=False)
                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', data=importance_df.head(15), ax=ax_imp)
                st.pyplot(fig_imp)

    elif page == "🩺 Make a Diagnosis":
        st.header("🩺 Get a Diagnosis with Explainable AI")
        st.markdown("Adjust the sliders to match the patient's tumor features and get a diagnosis.")
        
        model, scaler, _, _, _, feature_names, explainer = train_model(df, "Random Forest")

        input_data = {}
        mean_features = [f for f in feature_names if 'mean' in f]
        se_features = [f for f in feature_names if 'se' in f]
        worst_features = [f for f in feature_names if 'worst' in f]

        with st.expander("Mean Features", expanded=True):
            cols = st.columns(2)
            for i, feature in enumerate(mean_features):
                with cols[i % 2]:
                    input_data[feature] = st.slider(label=f"{feature}", min_value=float(df[feature].min()), max_value=float(df[feature].max()), value=float(df[feature].mean()), key=f"slider_{feature}")
        
        with st.expander("Standard Error Features"):
            cols = st.columns(2)
            for i, feature in enumerate(se_features):
                with cols[i % 2]:
                    input_data[feature] = st.slider(label=f"{feature}", min_value=float(df[feature].min()), max_value=float(df[feature].max()), value=float(df[feature].mean()), key=f"slider_{feature}")

        with st.expander("Worst Features"):
            cols = st.columns(2)
            for i, feature in enumerate(worst_features):
                with cols[i % 2]:
                    input_data[feature] = st.slider(label=f"{feature}", min_value=float(df[feature].min()), max_value=float(df[feature].max()), value=float(df[feature].mean()), key=f"slider_{feature}")

        if st.button("Get Diagnosis", type="primary"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df.values)
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)

            st.subheader("Diagnosis Result")
            if prediction[0] == 1:
                st.error(f"**Diagnosis: Malignant** (Probability: {prediction_proba[0][1]:.2f})")
            else:
                st.success(f"**Diagnosis: Benign** (Probability: {prediction_proba[0][0]:.2f})")

            st.subheader("Prediction Explanation (LIME)")
            with st.spinner('Generating explanation...'):
                explanation = explainer.explain_instance(
                    input_scaled[0],
                    model.predict_proba,
                    num_features=10,
                    top_labels=1
                )
                st.markdown("The plot below shows the top features that influenced this specific prediction.")
                fig = explanation.as_pyplot_figure(label=prediction[0])
                st.pyplot(fig)

    elif page == "🔬 Simulations & Visualizations":
        st.header("🔬 Simulations & Advanced Visualizations")

        st.subheader("Principal Component Analysis (PCA)")
        st.markdown("PCA is a technique to simplify complex data. Here, we've reduced all 30 features into just two 'Principal Components' to visualize the overall structure of the data.")
        
        X = df.drop(['id', 'diagnosis'], axis=1)
        X_scaled = StandardScaler().fit_transform(X.values)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC 1', 'PC 2'])
        pca_df['diagnosis'] = df['diagnosis'].map({0: 'Benign', 1: 'Malignant'})

        fig_pca = px.scatter(pca_df, x='PC 1', y='PC 2', color='diagnosis', title='2D PCA of Breast Cancer Data',
                             color_discrete_map={'Benign': 'green', 'Malignant': 'red'})
        st.plotly_chart(fig_pca, use_container_width=True)
        
        with st.expander("How to Interpret this Plot?"):
            explained_variance = pca.explained_variance_ratio_
            st.markdown(f"""
            - **The Goal:** This plot simplifies 30+ features into just two summary scores: Principal Component 1 (X-axis) and Principal Component 2 (Y-axis).
            - **PC1** captures **{explained_variance[0]:.1%}** of the total variance in the data. It primarily represents a combination of features related to tumor size and severity (like `radius_worst`, `perimeter_worst`, and `concave points_worst`).
            - **PC2** captures **{explained_variance[1]:.1%}** of the variance. It likely represents different aspects like cell texture and symmetry.
            - **The Key Insight:** Notice how well the **Malignant** (red) and **Benign** (green) tumors are separated into distinct groups. This clear separation is why a machine learning model can learn to accurately predict the diagnosis.
            """)

        st.subheader("Interactive 3D Scatter Plot")
        st.markdown("Select any three features to explore their relationships in a 3D space.")
        col3, col4, col5 = st.columns(3)
        feature_names_3d = df.drop(['id', 'diagnosis'], axis=1).columns
        with col3:
            x_axis_3d = st.selectbox("Select X-Axis for 3D", feature_names_3d, index=0, key="x_3d")
        with col4:
            y_axis_3d = st.selectbox("Select Y-Axis for 3D", feature_names_3d, index=1, key="y_3d")
        with col5:
            z_axis_3d = st.selectbox("Select Z-Axis for 3D", feature_names_3d, index=2, key="z_3d")
        
        df_plot_3d = df.copy()
        df_plot_3d['diagnosis'] = df_plot_3d['diagnosis'].map({0: 'Benign', 1: 'Malignant'})
        fig_3d = px.scatter_3d(df_plot_3d, x=x_axis_3d, y=y_axis_3d, z=z_axis_3d, color='diagnosis',
                               color_discrete_map={'Benign': 'green', 'Malignant': 'red'}, labels={'diagnosis': 'Diagnosis'},
                               title="3D View of Tumor Features")
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_3d, use_container_width=True)

if __name__ == "__main__":
    main()
