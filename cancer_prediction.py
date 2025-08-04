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
import time
import plotly.express as px
import lime
import lime.lime_tabular



@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the breast cancer dataset."""
    df = pd.read_csv(file_path)
    if 'Unnamed: 32' in df.columns:
        df = df.drop('Unnamed: 32', axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

@st.cache_resource
def train_model(df, model_name):
    """Trains a selected model and returns the model, scaler, and LIME explainer."""
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
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


def main():
    st.set_page_config(page_title="Breast Cancer Diagnosis", layout="wide")
    st.title("Advanced Breast Cancer Diagnosis AI")
    st.markdown("An advanced application for diagnosis, model comparison, and explainable AI.")

    try:
        df = load_data("Breast_cancer_dataset.csv")
    except FileNotFoundError:
        st.error("The `Breast_cancer_dataset.csv` file was not found. Please make sure it's in the same directory.")
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Exploratory Data Analysis", "Model Performance", "Make a Diagnosis", "Simulations & Visualizations"])

    if page == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis (EDA)")
        if st.checkbox("Show Raw Data"):
            st.write(df)
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Diagnosis Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='diagnosis', data=df, ax=ax)
            ax.set_xticklabels(['Benign (0)', 'Malignant (1)'])
            st.pyplot(fig)
        with col2:
            st.subheader("Feature Distributions")
            feature = st.selectbox("Select a feature:", df.drop(['id', 'diagnosis'], axis=1).columns)
            fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
            sns.histplot(data=df, x=feature, hue='diagnosis', kde=True, ax=ax_dist)
            plt.legend(title='Diagnosis', labels=['Malignant', 'Benign'])
            st.pyplot(fig_dist)
        st.subheader("Feature Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(18, 14))
        sns.heatmap(df.drop(['id'], axis=1).corr(), annot=True, fmt='.1f', cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

    elif page == "Model Performance":
        st.header("Machine Learning Model Performance")
        model_name = st.selectbox("Select a Model", ["Logistic Regression", "Random Forest", "Support Vector Machine"])
        
        model, _, accuracy, conf_matrix, class_report, feature_names, _ = train_model(df, model_name)
        
        st.write(f"**Model:** {model_name}")
        st.write(f"**Accuracy:** {accuracy:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)
        with col2:
            st.subheader("Classification Report")
            st.table(pd.DataFrame(class_report).transpose())

        if model_name == "Random Forest":
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
            importance_df = importance_df.sort_values('importance', ascending=False)
            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=importance_df.head(15), ax=ax_imp)
            st.pyplot(fig_imp)

    elif page == "Make a Diagnosis":
        st.header("Get a Diagnosis with Explainable AI")
        st.markdown("Adjust the sliders and see how each feature influences the diagnosis.")
        
        model, scaler, _, _, _, feature_names, explainer = train_model(df, "Random Forest")

        input_data = {}
        mean_features = [f for f in feature_names if 'mean' in f]
        se_features = [f for f in feature_names if 'se' in f]
        worst_features = [f for f in feature_names if 'worst' in f]

        with st.expander("Mean Features"):
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

        if st.button("Get Diagnosis"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
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
                    num_features=10
                )
                fig = explanation.as_pyplot_figure()
                st.pyplot(fig)


    elif page == "Simulations & Visualizations":
        st.header("Simulations & Advanced Visualizations")

        st.subheader("Principal Component Analysis (PCA)")
        st.markdown("PCA helps visualize the high-dimensional data in 2D, showing how well the classes are separated.")
        
        X = df.drop(['id', 'diagnosis'], axis=1)
        X_scaled = StandardScaler().fit_transform(X.values)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC 1', 'PC 2'])
        pca_df['diagnosis'] = df['diagnosis']

        fig_pca = px.scatter(pca_df, x='PC 1', y='PC 2', color='diagnosis', title='2D PCA of Breast Cancer Data',
                             color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig_pca, use_container_width=True)

        st.subheader("Interactive 3D Scatter Plot")
        col3, col4, col5 = st.columns(3)
        feature_names_3d = df.drop(['id', 'diagnosis'], axis=1).columns
        with col3:
            x_axis_3d = st.selectbox("Select X-Axis for 3D", feature_names_3d, index=0, key="x_3d")
        with col4:
            y_axis_3d = st.selectbox("Select Y-Axis for 3D", feature_names_3d, index=1, key="y_3d")
        with col5:
            z_axis_3d = st.selectbox("Select Z-Axis for 3D", feature_names_3d, index=2, key="z_3d")
        
        fig_3d = px.scatter_3d(df, x=x_axis_3d, y=y_axis_3d, z=z_axis_3d, color='diagnosis',
                               color_discrete_map={0: 'green', 1: 'red'}, labels={'diagnosis': 'Diagnosis'},
                               title="3D View of Tumor Features")
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_3d, use_container_width=True)

if __name__ == "__main__":
    main()
