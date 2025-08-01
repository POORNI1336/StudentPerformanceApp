# app.py

# -*- coding: utf-8 -*-
"""
Streamlit Web App for Student Performance Prediction and Dashboard
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Constants
MODEL_FILENAME = 'trained_model.sav'
DATA_FILENAME = 'StudentsPerformance.csv'

# -------------------- Data Loading & Processing -------------------- #
@st.cache_data
def load_and_process_data():
    """Load dataset and preprocess it for model training."""
    try:
        df = pd.read_csv(DATA_FILENAME)
    except FileNotFoundError:
        st.error(f"'{DATA_FILENAME}' not found. Please ensure the file is in the directory.")
        return None

    df['average score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    df['Outcome'] = df['average score'].apply(lambda x: 1 if x >= 60 else 0)
    return df

# -------------------- Model Training -------------------- #
@st.cache_resource
def train_and_save_model(df):
    """Train an SVM classifier and save the model."""
    if df is None:
        return None, None

    X = df[['math score', 'reading score', 'writing score']]
    Y = df['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    model = svm.SVC(kernel='linear')
    model.fit(X_train, Y_train)

    try:
        pickle.dump(model, open(MODEL_FILENAME, 'wb'))
        st.success(f"Model saved as '{MODEL_FILENAME}'.")
    except Exception as e:
        st.error(f"Failed to save model: {e}")
        return None, None

    test_acc = accuracy_score(model.predict(X_test), Y_test)
    train_acc = accuracy_score(model.predict(X_train), Y_train)
    st.info(f"Training Accuracy: {train_acc:.2f}")
    st.info(f"Test Accuracy: {test_acc:.2f}")

    return model, test_acc

# -------------------- Prediction Function -------------------- #
def predict_performance(scores):
    """Return prediction result based on input scores."""
    if loaded_model is None:
        return "Model not loaded."
    try:
        data = np.asarray(scores, dtype=float).reshape(1, -1)
        return 'The student passed.' if loaded_model.predict(data)[0] == 1 else 'The student failed.'
    except Exception as e:
        return f"Error during prediction: {e}"

# -------------------- Dashboard Visualizations -------------------- #
def show_dashboard(df):
    st.header("Student Performance Dashboard")

    if df is None:
        st.warning("Data unavailable for dashboard.")
        return

    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.subheader("Score Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, column, color in zip(axes, ['math score', 'reading score', 'writing score'], ['skyblue', 'lightcoral', 'lightgreen']):
        sns.histplot(df[column], kde=True, ax=ax, color=color)
        ax.set_title(f'{column.title()} Distribution')
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Outcome by Gender")
        gender_stats = df.groupby('gender')['Outcome'].value_counts(normalize=True).unstack()
        st.dataframe(gender_stats)

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='gender', hue='Outcome', palette='viridis', ax=ax1)
        ax1.set_title("Outcome by Gender")
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.markdown("#### Outcome by Test Preparation")
        prep_stats = df.groupby('test preparation course')['Outcome'].value_counts(normalize=True).unstack()
        st.dataframe(prep_stats)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='test preparation course', hue='Outcome', palette='magma', ax=ax2)
        ax2.set_title("Outcome by Test Preparation")
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("Average Scores by Parental Education")
    edu_avg = df.groupby('parental level of education')[['math score', 'reading score', 'writing score']].mean().reset_index()
    st.dataframe(edu_avg)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    edu_avg.set_index('parental level of education').plot(kind='bar', colormap='Paired', ax=ax3)
    ax3.set_title("Avg Scores by Parental Education")
    ax3.set_ylabel("Average Score")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig3)
    plt.close(fig3)

    st.subheader("Correlation Heatmap")
    corr = df[['math score', 'reading score', 'writing score', 'average score', 'Outcome']].corr()
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
    ax4.set_title("Correlation Heatmap")
    st.pyplot(fig4)
    plt.close(fig4)

# -------------------- Streamlit App -------------------- #
def main():
    st.set_page_config(layout="wide")
    st.title("🎓 Student Performance Prediction & Dashboard")

    tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Dashboard"])

    with tab1:
        st.header("Performance Prediction")
        st.write("Enter student's scores to predict if they passed (average score ≥ 60).")

        math = st.number_input("Math Score", 0, 100, 70)
        reading = st.number_input("Reading Score", 0, 100, 75)
        writing = st.number_input("Writing Score", 0, 100, 72)

        if st.button("Predict Performance"):
            result = predict_performance([math, reading, writing])
            st.success(result)

        if model_accuracy is not None:
            st.markdown("---")
            st.info(f"Model Test Accuracy: **{model_accuracy:.2f}**")

    with tab2:
        show_dashboard(student_data)

# -------------------- Main Execution -------------------- #
student_data = load_and_process_data()
loaded_model, model_accuracy = train_and_save_model(student_data)

if __name__ == "__main__":
    main()
