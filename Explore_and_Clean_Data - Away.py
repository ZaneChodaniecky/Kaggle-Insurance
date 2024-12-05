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
from pandas.tseries.offsets import DateOffset



if socket.gethostname() == 'zchodani-p-l01':
    file_directory = r"C:\Users\zchodaniecky\OneDrive - Franklin Templeton\Documents\Python\Kaggle\Insurance Premiums"
elif socket.gethostname() == 'FTILC3VBil7BwCe':
    file_directory = r"C:\Users\zchodan\OneDrive - Franklin Templeton\Documents\Python\Kaggle\Insurance Premiums"
else:
    file_directory = r"C:\Users\zanec\OneDrive\Documents\Python\Kaggle\Playground Series\s4e12 - Insurance Premiums"
         
os.chdir(file_directory)

# Load the Data
df = pd.read_csv('Insurance Premium Prediction Dataset.csv')


# Inspect the Data
df.head()
df.info()
df.describe(include='all')
print(df.shape)
print(df.columns)

df['Annual Income'].describe()
df['Policy Start Date'].head()

# Identify duplicate rows
duplicates = df.duplicated().sum()
df.drop_duplicates(inplace=True)

# Identify null data
print (df.isnull().sum())
print (round((df.isnull().sum() / len(df)) * 100),1)

# Drop rows with missing fields that are less than 5% of data and not easily estimated 
df = df.dropna(subset=['Age','Marital Status','Health Score','Vehicle Age','Insurance Duration', 'Premium Amount'])


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

### Find average health score based on exercise frequency and smoking status and fill null values with it
average_health_score = df.groupby(['Exercise Frequency','Smoking Status'])['Health Score'].mean().reset_index()
print(average_health_score)
# Filled with the average of 2
df['Number of Dependents'] = df['Number of Dependents'].fillna(2)

### Find average number of claims based on age category, and fill null values with it
average_previous_claims = df.groupby(['Age Category'])['Previous Claims'].mean().reset_index()
print(average_previous_claims)
# Filled with the average of 2
df['Previous Claims'] = df['Previous Claims'].fillna(1)

### Find average credit score based on age category fill null values with it
average_credit_score = df.groupby(['Age Category'])['Credit Score'].mean().reset_index()
print(average_credit_score)
# Filled with the average of 2
df['Credit Score'] = df['Credit Score'].fillna(575)

# Convert and Extract date from Policy Start Date
df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])

df['Policy Start Year'] = df['Policy Start Date'].dt.year
df['Policy Start Month'] = df['Policy Start Date'].dt.month
df['Policy Start Weekday'] = df['Policy Start Date'].dt.dayofweek
df['Policy Start Month sin'] = np.sin(2 * np.pi * df['Policy Start Month'] / 12)
df['Policy Start Month cos'] = np.cos(2 * np.pi * df['Policy Start Month'] / 12)

# Drop Policy Start Date and keep its transformation fields
df = df.drop(columns='Policy Start Date')


# Create Policy End Date field
#df['Policy End Date'] = df['Policy Start Date'] + df['Insurance Duration'].apply(lambda x: DateOffset(years=x))
#df['Policy End Date'] = pd.to_datetime(df['Policy End Date'])
#df['Policy End Date'] = df['Policy End Date'].dt.date


# Handle Categorical Null Data
df['Occupation'] = df['Occupation'].fillna('Unknown')
df['Customer Feedback'] = df['Customer Feedback'].fillna('Unknown')

# Encode Categorical Data (sloppy, can probably delete it is recreated below better)
# =============================================================================
# df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True).astype(int)
# df_encoded = pd.get_dummies(df_encoded, columns=['Marital Status'], drop_first=True).astype(int)
# df_encoded = pd.get_dummies(df_encoded, columns=['Education Level'], drop_first=True).astype(int)
# df_encoded = pd.get_dummies(df_encoded, columns=['Occupation'], drop_first=True).astype(int)
# df_encoded = pd.get_dummies(df_encoded, columns=['Location'], drop_first=True).astype(int)
# df_encoded = pd.get_dummies(df_encoded, columns=['Policy Type'], drop_first=True).astype(int)
# df_encoded = pd.get_dummies(df_encoded, columns=['Customer Feedback'], drop_first=True).astype(int)
# df_encoded = pd.get_dummies(df_encoded, columns=['Smoking Status'], drop_first=True).astype(int)
# df_encoded = pd.get_dummies(df_encoded, columns=['Exercise Frequency'], drop_first=True).astype(int)
# df_encoded = pd.get_dummies(df_encoded, columns=['Property Type'], drop_first=True).astype(int)
# 
# df_encoded = pd.get_dummies(df_encoded, columns=['Age Category'], drop_first=True).astype(int)
# =============================================================================


# List of columns to encode
columns_to_encode = [
    'Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location',
    'Policy Type', 'Customer Feedback', 'Smoking Status', 'Exercise Frequency',
    'Property Type', 'Age Category'
]

# Apply pd.get_dummies to all specified columns, dropping the first category for each and convert to binary int
df_encoded = df.copy()
for col in columns_to_encode:
    # Get the column names before encoding
    existing_columns = set(df_encoded.columns)
    
    # Create dummy columns
    df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)

    # Identify the newly created columns
    new_columns = set(df_encoded.columns) - existing_columns
    
    # Convert the new columns to integers
    df_encoded[list(new_columns)] = df_encoded[list(new_columns)].astype(int)



# Identify null data in new transformed dataset
print(df_encoded.isnull().sum())
print(round((df_encoded.isnull().sum() / len(df_encoded)) * 100),1)

# Check that everything is a numerical column
df_encoded.info()

# Check that there are no nulls
df_encoded.isna().sum()

# List of columns to convert to int
columns_to_convert = [
    'Age', 'Annual Income', 'Number of Dependents', 
    'Previous Claims', 'Vehicle Age', 'Credit Score', 
    'Insurance Duration'
]

# Convert the specified columns to int
df_encoded[columns_to_convert] = df_encoded[columns_to_convert].astype(int)

## Delete out column with names LIKE
# word = 'Smoking'
# matching_columns = [col for col in df_encoded.columns if word.lower() in col.lower()]
# df_encoded = df_encoded.drop(columns=matching_columns,axis=1)

df_encoded.to_csv('Transformed.csv',index=False)
















