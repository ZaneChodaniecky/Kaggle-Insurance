# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:30:41 2024

@author: zchodan
"""

import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import os
import itertools
import socket
from datetime import datetime




if socket.gethostname() == 'zchodani-p-l01':
    file_directory = r"C:\Users\zchodaniecky\OneDrive - Franklin Templeton\Documents\Python\NHL_data"
elif socket.gethostname() == 'FTILC3VBil7BwCe':
    file_directory = r"C:\Users\zchodan\OneDrive - Franklin Templeton\Documents\Python\Kaggle\Insurance Premiums"
else:
    file_directory = r"C:\Users\zanec\OneDrive\Documents\Python\NHL_data"
         
os.chdir(file_directory)

# Load the Data
df = pd.read_csv('Insurance Premium Prediction Dataset.csv')

# Inspect the Data
df.head()
df.info()
df.describe(include='all')
print(df.shape)
print(df.columns)

# Identify duplicate rows
duplicates = df.duplicated().sum()
#df.drop_duplicates(inplace=True)

# Identify null data
print (round((df.isnull().sum() / len(df)) * 100),1)

# Drop rows with missing fields that are less than 5% of data and not easily estimated [Also premium since thats the independent variable]
df = df.dropna(subset=['Age','Marital Status','Health Score','Premium Amount'])


# Create categories for the Age field
df['Age'].describe()
bins = [17,28,37,46,55,65]
labels = ['17-28','29-37','38-46','47-55','55-65']
df['Age Category'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)


# Handle Low Cardinality Categorical Data
df['Gender'].value_counts()
df['Marital Status'].value_counts() # has 2% NA
df['Education Level'].value_counts()
df['Occupation'].value_counts() # has 29 %NA
df['Location'].value_counts()
df['Policy Type'].value_counts()
df['Customer Feedback'].value_counts()
df['Smoking Status'].value_counts()
df['Exercise Frequency'].value_counts()
df['Property Type'].value_counts()

df['Age Category'].value_counts()



### Find average income based on age category and education and fill null values with it
average_income = df.groupby(['Age Category', 'Education Level'])['Annual Income'].mean().reset_index()
# Step 2: Merge the calculated means back into the original DataFrame
df = df.merge(average_income, on=['Age Category', 'Education Level'], how='left', suffixes=('', '_mean'))
# Step 3: Fill NaN values in 'Annual Income' using the corresponding mean values
df['Annual Income'] = df['Annual Income'].fillna(df['Annual Income_mean'])
# Step 4: Drop the temporary mean column
df = df.drop(columns=['Annual Income_mean'])

### Find average number of dependents based on age category, education, and property type and fill null values with it
average_dependents = df.groupby(['Education Level'])['Number of Dependents'].mean().reset_index()
print(average_dependents)
# Filled with the average of 2
df['Number of Dependents'] = df['Number of Dependents'].fillna(2)

### Find average health score based on age category, Marital Status,, and fill null values with it
average_health_score = df.groupby(['Exercise Frequency','Smoking Status'])['Health Score'].mean().reset_index()
print(average_health_score)
# Filled with the average of 2
df['Number of Dependents'] = df['Number of Dependents'].fillna(2)

### Find average health score based on age category, Marital Status,, and fill null values with it
average_previous_claims = df.groupby(['Age Category'])['Previous Claims'].mean().reset_index()
print(average_previous_claims)
# Filled with the average of 2
df['Previous Claims'] = df['Previous Claims'].fillna(1)

### Find average health score based on age category, Marital Status,, and fill null values with it
average_credit_score = df.groupby(['Age Category'])['Credit Score'].mean().reset_index()
print(average_credit_score)
# Filled with the average of 2
df['Credit Score'] = df['Credit Score'].fillna(575)



# Handle Categorical Null Data
df['Occupation'] = df['Occupation'].fillna('Unknown')
df['Customer Feedback'] = df['Customer Feedback'].fillna('Unknown')

# Encode Categorical Data
df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)
df_encoded = pd.get_dummies(df_encoded, columns=['Marital Status'], drop_first=True)
df_encoded = pd.get_dummies(df_encoded, columns=['Education Level'], drop_first=True)
df_encoded = pd.get_dummies(df_encoded, columns=['Occupation'], drop_first=True)
df_encoded = pd.get_dummies(df_encoded, columns=['Location'], drop_first=True)
df_encoded = pd.get_dummies(df_encoded, columns=['Policy Type'], drop_first=True)
df_encoded = pd.get_dummies(df_encoded, columns=['Customer Feedback'], drop_first=True)
df_encoded = pd.get_dummies(df_encoded, columns=['Smoking Status'], drop_first=True)
df_encoded = pd.get_dummies(df_encoded, columns=['Exercise Frequency'], drop_first=True)
df_encoded = pd.get_dummies(df_encoded, columns=['Property Type'], drop_first=True)

df_encoded = pd.get_dummies(df_encoded, columns=['Age Category'], drop_first=True)


# Identify null data in new transformed dataset
print (round((df_encoded.isnull().sum() / len(df_encoded)) * 100),1)




# Convert Data Types
df['Age'] = df['Age'].astype(int) # Has NAs
df['Gender'] = df['Age'].astype(int) # Has NAs



df['column'] = pd.to_datetime(df['column'])  # For dates
df['column'] = df['column'].astype(int)  # For numeric conversions












corr_matrix = df.corr(numeric_only=True)
corr_matrix['Credit Score'].sort_values(ascending=False)









