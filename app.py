# app.py

# -*- coding: utf-8 -*-
"""
Streamlit Web App for Student Performance Prediction and Dashboard
"""

import numpy as np
import pandas as pd
import pickle
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define the filename for the trained model and data
MODEL_FILENAME = 'trained_model.sav'
DATA_FILENAME = 'StudentsPerformance.csv'

@st.cache_data # Changed to st.cache_data as it's for data loading/processing
def load_and_process_data():
    """
    Loads the student performance data, calculates average score, and defines outcome.
    """
    try:
        student_dataset = pd.read_csv(DATA_FILENAME)
    except FileNotFoundError:
        st.error(f"Error: '{DATA_FILENAME}' not found. Please ensure the file is in the same directory.")
        return None

    # Calculate 'average score' and define 'Outcome' (1 for Pass >= 60, 0 for Fail < 60)
    student_dataset['average score'] = student_dataset[['math score', 'reading score', 'writing score']].mean(axis=1)
    student_dataset['Outcome'] = student_dataset['average score'].apply(lambda x: 1 if x >= 60 else 0)
    return student_dataset

@st.cache_resource
def train_and_save_model(student_dataset):
    """
    Trains an SVM model for student performance prediction and saves it.
    If the model file already exists, it will be overwritten to ensure
    the model is trained on the student performance data as requested.
    """
    if student_dataset is None:
        return None, None

    st.write("Training model... This might take a moment.")

    # Select features (X) and target (Y)
    X = student_dataset[['math score', 'reading score', 'writing score']]
    Y = student_dataset['Outcome']

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Train the Support Vector Machine (SVM) classifier
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)

    # Evaluate model performance
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    # Save the trained model
    try:
        pickle.dump(classifier, open(MODEL_FILENAME, 'wb'))
        st.success(f"Model trained and saved as '{MODEL_FILENAME}'.")
        st.info(f"Training Data Accuracy: {training_data_accuracy:.2f}")
        st.info(f"Test Data Accuracy: {test_data_accuracy:.2f}")
    except Exception as e:
        st.error(f"Error saving model: {e}")
        return None, None

    return classifier, test_data_accuracy

# Load and process data first
student_data = load_and_process_data()

# Train and save the model using the processed data
loaded_model, model_accuracy = train_and_save_model(student_data)

# Prediction function for student performance
def student_performance_prediction(input_data):
    """
    Predicts whether a student passed or failed based on input scores.

    Args:
        input_data (list): A list containing the math score, reading score, and writing score.

    Returns:
        str: A message indicating whether the student passed or failed.
    """
    # Ensure the model is loaded before attempting prediction
    if loaded_model is None:
        return "Model not loaded. Cannot make predictions."

    # Convert input list to a NumPy array with float data type
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the array as we are predicting for a single instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make the prediction using the loaded model
    prediction = loaded_model.predict(input_data_reshaped)

    # Interpret the prediction
    if prediction[0] == 0:
        return 'The student failed.'
    else:
        return 'The student passed.'

# Streamlit Web App Interface
def main():
    """
    Main function to define the Streamlit web application layout and logic.
    """
    st.set_page_config(layout="wide") # Use wide layout for dashboard

    st.title('Student Performance Analysis and Prediction Web App')

    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Prediction", "Dashboard"])

    with tab1:
        st.header('Student Performance Prediction')
        st.write("Enter the student's scores to predict whether they passed or failed (based on an average score of 60 or higher):")

        # Input fields for student scores
        math_score = st.number_input('Math Score (0-100)', min_value=0, max_value=100, value=70, step=1)
        reading_score = st.number_input('Reading Score (0-100)', min_value=0, max_value=100, value=75, step=1)
        writing_score = st.number_input('Writing Score (0-100)', min_value=0, max_value=100, value=72, step=1)

        prediction_result = ''

        if st.button('Predict Performance'):
            if loaded_model is None:
                st.error("Model not loaded. Please ensure the data file is present and try again.")
                return

            try:
                input_list = [math_score, reading_score, writing_score]
                prediction_result = student_performance_prediction(input_list)
            except ValueError:
                prediction_result = "Please enter valid numerical values for all scores."
            except Exception as e:
                prediction_result = f"An error occurred during prediction: {e}"

        st.success(prediction_result)

        if model_accuracy is not None:
            st.markdown(f"---")
            st.info(f"Model Test Accuracy: {model_accuracy:.2f}")

    with tab2:
        st.header('Student Performance Dashboard')
        if student_data is None:
            st.warning("Cannot display dashboard. Data file not found or could not be processed.")
            return

        st.markdown("### Data Overview")
        st.dataframe(student_data.head())

        st.markdown("### Distribution of Scores")
        # Plotting distribution of math, reading, and writing scores
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(student_data['math score'], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title('Math Score Distribution')
        sns.histplot(student_data['reading score'], kde=True, ax=axes[1], color='lightcoral')
        axes[1].set_title('Reading Score Distribution')
        sns.histplot(student_data['writing score'], kde=True, ax=axes[2], color='lightgreen')
        axes[2].set_title('Writing Score Distribution')
        st.pyplot(fig)
        plt.close(fig) # Close the figure to free up memory

        st.markdown("### Performance by Gender and Test Preparation Course")
        col1, col2 = st.columns(2)

        with col1:
            # Performance by Gender
            gender_performance = student_data.groupby('gender')['Outcome'].value_counts(normalize=True).unstack()
            st.write("#### Outcome by Gender")
            st.dataframe(gender_performance)
            fig_gender, ax_gender = plt.subplots(figsize=(6, 4))
            sns.countplot(data=student_data, x='gender', hue='Outcome', palette='viridis', ax=ax_gender)
            ax_gender.set_title('Student Outcome by Gender')
            ax_gender.set_xlabel('Gender')
            ax_gender.set_ylabel('Number of Students')
            st.pyplot(fig_gender)
            plt.close(fig_gender)

        with col2:
            # Performance by Test Preparation Course
            test_prep_performance = student_data.groupby('test preparation course')['Outcome'].value_counts(normalize=True).unstack()
            st.write("#### Outcome by Test Preparation Course")
            st.dataframe(test_prep_performance)
            fig_test_prep, ax_test_prep = plt.subplots(figsize=(6, 4))
            sns.countplot(data=student_data, x='test preparation course', hue='Outcome', palette='magma', ax=ax_test_prep)
            ax_test_prep.set_title('Student Outcome by Test Preparation Course')
            ax_test_prep.set_xlabel('Test Preparation Course')
            ax_test_prep.set_ylabel('Number of Students')
            st.pyplot(fig_test_prep)
            plt.close(fig_test_prep)

        st.markdown("### Average Scores by Parental Level of Education")
        # Average scores by parental level of education
        parental_education_scores = student_data.groupby('parental level of education')[['math score', 'reading score', 'writing score']].mean().reset_index()
        st.dataframe(parental_education_scores)

        fig_parental_edu, ax_parental_edu = plt.subplots(figsize=(10, 6))
        parental_education_scores.set_index('parental level of education').plot(kind='bar', ax=ax_parental_edu, colormap='Paired')
        ax_parental_edu.set_title('Average Scores by Parental Level of Education')
        ax_parental_edu.set_ylabel('Average Score')
        ax_parental_edu.set_xlabel('Parental Level of Education')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig_parental_edu)
        plt.close(fig_parental_edu)

        st.markdown("### Correlation Heatmap of Scores")
        # Correlation heatmap
        numeric_data = student_data[['math score', 'reading score', 'writing score', 'average score', 'Outcome']]
        correlation_matrix = numeric_data.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        ax_corr.set_title('Correlation Matrix of Scores and Outcome')
        st.pyplot(fig_corr)
        plt.close(fig_corr)


if __name__ == '__main__':
    main()
