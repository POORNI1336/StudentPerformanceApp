import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

st.title("Student Performance Dashboard")

# Load data with caching
@st.cache_data
def load_data():
    # Ensure the correct CSV file name is used
    return pd.read_csv("StudentsPerformance.csv")

df = load_data()

# Display raw data
st.subheader("Raw Data")
st.dataframe(df)

st.sidebar.header("Filter Options")

# Function to filter data based on sidebar selections
def filter_data(df):
    # Numerical filters for scores
    math_score = st.sidebar.slider(
        "Math Score",
        int(df['math score'].min()),
        int(df['math score'].max()),
        (int(df['math score'].min()), int(df['math score'].max()))
    )
    reading_score = st.sidebar.slider(
        "Reading Score",
        int(df['reading score'].min()),
        int(df['reading score'].max()),
        (int(df['reading score'].min()), int(df['reading score'].max()))
    )
    writing_score = st.sidebar.slider(
        "Writing Score",
        int(df['writing score'].min()),
        int(df['writing score'].max()),
        (int(df['writing score'].min()), int(df['writing score'].max()))
    )

    # Categorical filters
    selected_gender = st.sidebar.multiselect(
        "Select Gender",
        options=df['gender'].unique(),
        default=df['gender'].unique()
    )
    selected_test_prep = st.sidebar.multiselect(
        "Select Test Preparation Course",
        options=df['test preparation course'].unique(),
        default=df['test preparation course'].unique()
    )
    selected_parental_education = st.sidebar.multiselect(
        "Select Parental Level of Education",
        options=df['parental level of education'].unique(),
        default=df['parental level of education'].unique()
    )
    selected_lunch = st.sidebar.multiselect(
        "Select Lunch Type",
        options=df['lunch'].unique(),
        default=df['lunch'].unique()
    )
    selected_race_ethnicity = st.sidebar.multiselect(
        "Select Race/Ethnicity",
        options=df['race/ethnicity'].unique(),
        default=df['race/ethnicity'].unique()
    )


    filtered_df = df[
        (df['math score'] >= math_score[0]) & (df['math score'] <= math_score[1]) &
        (df['reading score'] >= reading_score[0]) & (df['reading score'] <= reading_score[1]) &
        (df['writing score'] >= writing_score[0]) & (df['writing score'] <= writing_score[1]) &
        (df['gender'].isin(selected_gender)) &
        (df['test preparation course'].isin(selected_test_prep)) &
        (df['parental level of education'].isin(selected_parental_education)) &
        (df['lunch'].isin(selected_lunch)) &
        (df['race/ethnicity'].isin(selected_race_ethnicity))
    ]
    return filtered_df

filtered_df = filter_data(df)

# Display filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_df)

st.subheader("Data Visualizations")

# 1. Scatter Plot: Reading Score vs. Math Score (Hue by Gender)
st.write("### Reading Score vs. Math Score (by Gender)")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=filtered_df, x='math score', y='reading score', hue='gender', ax=ax, palette='viridis')
ax.set_title('Reading Score vs. Math Score')
ax.set_xlabel('Math Score')
ax.set_ylabel('Reading Score')
st.pyplot(fig)

# 2. Histogram: Math Score Distribution
st.write("### Math Score Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(filtered_df['math score'], kde=True, ax=ax, color='skyblue')
ax.set_title('Distribution of Math Scores')
ax.set_xlabel('Math Score')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# 3. Histogram: Reading Score Distribution
st.write("### Reading Score Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(filtered_df['reading score'], kde=True, ax=ax, color='lightcoral')
ax.set_title('Distribution of Reading Scores')
ax.set_xlabel('Reading Score')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# 4. Histogram: Writing Score Distribution
st.write("### Writing Score Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(filtered_df['writing score'], kde=True, ax=ax, color='lightgreen')
ax.set_title('Distribution of Writing Scores')
ax.set_xlabel('Writing Score')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# 5. Correlation Heatmap: Scores
st.write("### Correlation Heatmap of Scores")
# Select only numerical score columns for correlation
score_columns = ['math score', 'reading score', 'writing score']
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(filtered_df[score_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title('Correlation Matrix of Student Scores')
st.pyplot(fig)

# 6. Count Plot: Gender Distribution
st.write("### Gender Distribution")
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=filtered_df, x='gender', ax=ax, palette='pastel')
ax.set_title('Distribution of Gender')
ax.set_xlabel('Gender')
ax.set_ylabel('Count')
st.pyplot(fig)

# 7. Count Plot: Test Preparation Course Distribution
st.write("### Test Preparation Course Distribution")
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=filtered_df, x='test preparation course', ax=ax, palette='muted')
ax.set_title('Distribution of Test Preparation Course Completion')
ax.set_xlabel('Test Preparation Course')
ax.set_ylabel('Count')
st.pyplot(fig)

# 8. Boxplot: Math Score by Test Preparation Course
st.write("### Math Score by Test Preparation Course")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=filtered_df, x='test preparation course', y='math score', ax=ax, palette='coolwarm')
ax.set_title('Math Score Distribution by Test Preparation Course')
ax.set_xlabel('Test Preparation Course')
ax.set_ylabel('Math Score')
st.pyplot(fig)

# 9. Boxplot: Reading Score by Test Preparation Course
st.write("### Reading Score by Test Preparation Course")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=filtered_df, x='test preparation course', y='reading score', ax=ax, palette='coolwarm')
ax.set_title('Reading Score Distribution by Test Preparation Course')
ax.set_xlabel('Test Preparation Course')
ax.set_ylabel('Reading Score')
st.pyplot(fig)

# 10. Boxplot: Writing Score by Test Preparation Course
st.write("### Writing Score by Test Preparation Course")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=filtered_df, x='test preparation course', y='writing score', ax=ax, palette='coolwarm')
ax.set_title('Writing Score Distribution by Test Preparation Course')
ax.set_xlabel('Test Preparation Course')
ax.set_ylabel('Writing Score')
st.pyplot(fig)

# 11. Pairplot: All Scores (Hue by Test Preparation Course)
st.write("### Pairplot of Scores (by Test Preparation Course)")
# Ensure there's enough data for pairplot after filtering
if not filtered_df.empty:
    fig = sns.pairplot(filtered_df, vars=['math score', 'reading score', 'writing score'],
                       hue='test preparation course', palette='deep')
    st.pyplot(fig)
else:
    st.info("No data available for Pairplot after filtering.")

# 12. Violin Plot: Math Score by Parental Level of Education
st.write("### Math Score by Parental Level of Education (Violin Plot)")
fig, ax = plt.subplots(figsize=(12, 7))
sns.violinplot(data=filtered_df, x='parental level of education', y='math score', ax=ax, palette='tab10')
ax.set_title('Math Score Distribution by Parental Level of Education')
ax.set_xlabel('Parental Level of Education')
ax.set_ylabel('Math Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)

# 13. Bar Chart: Average Scores by Parental Level of Education
st.write("### Average Scores by Parental Level of Education")
avg_scores_by_parental_edu = filtered_df.groupby('parental level of education')[['math score', 'reading score', 'writing score']].mean().reset_index()
fig, ax = plt.subplots(figsize=(12, 7))
avg_scores_by_parental_edu.set_index('parental level of education').plot(kind='bar', ax=ax, colormap='Paired')
ax.set_title('Average Scores by Parental Level of Education')
ax.set_xlabel('Parental Level of Education')
ax.set_ylabel('Average Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)

# 14. Boxplot: Scores by Lunch Type
st.write("### Scores by Lunch Type")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
sns.boxplot(data=filtered_df, x='lunch', y='math score', ax=axes[0], palette='viridis')
axes[0].set_title('Math Score by Lunch Type')
sns.boxplot(data=filtered_df, x='lunch', y='reading score', ax=axes[1], palette='viridis')
axes[1].set_title('Reading Score by Lunch Type')
sns.boxplot(data=filtered_df, x='lunch', y='writing score', ax=axes[2], palette='viridis')
axes[2].set_title('Writing Score by Lunch Type')
plt.tight_layout()
st.pyplot(fig)

# 15. Bar Chart: Average Scores by Race/Ethnicity
st.write("### Average Scores by Race/Ethnicity")
avg_scores_by_race = filtered_df.groupby('race/ethnicity')[['math score', 'reading score', 'writing score']].mean().reset_index()
fig, ax = plt.subplots(figsize=(12, 7))
avg_scores_by_race.set_index('race/ethnicity').plot(kind='bar', ax=ax, colormap='Accent')
ax.set_title('Average Scores by Race/Ethnicity')
ax.set_xlabel('Race/Ethnicity')
ax.set_ylabel('Average Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)
