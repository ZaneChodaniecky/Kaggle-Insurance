# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:25:46 2024

@author: ZaneC
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
df = pd.read_csv('test.csv')


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
#df = df.dropna(subset=['Age','Vehicle Age','Insurance Duration'])


df.info()

# Create categories for the Age field
df['Age'].describe()
bins = [17,28,37,46,55,65]
labels = ['17-28','29-37','38-46','47-55','55-65']
df['Age Category'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)


# # Handle Low Cardinality Categorical Data
# df['Gender'].value_counts()
# df['Marital Status'].value_counts() # has 2% NA
# df['Education Level'].value_counts()
# df['Occupation'].value_counts() # has 29 %NA
# df['Location'].value_counts()
# df['Policy Type'].value_counts()
# df['Customer Feedback'].value_counts()
# df['Smoking Status'].value_counts()
# df['Exercise Frequency'].value_counts()
# df['Property Type'].value_counts()
# df['Age Category'].value_counts()

### Find average income based on age category and education and fill null values with it
# average_income = df.groupby(['Age Category', 'Education Level'])['Annual Income'].mean().reset_index()
# # Step 2: Merge the calculated means back into the original DataFrame
# df = df.merge(average_income, on=['Age Category', 'Education Level'], how='left', suffixes=('', '_mean'))
# # Step 3: Fill NaN values in 'Annual Income' using the corresponding mean values
# df['Annual Income'] = df['Annual Income'].fillna(df['Annual Income_mean'])
# # Step 4: Drop the temporary mean column
# df = df.drop(columns=['Annual Income_mean'])
average_annual_income = df.groupby(['Age Category'])['Annual Income'].mean().reset_index()
print(average_annual_income)
# Filled with the average of 32800
df['Annual Income'] = df['Annual Income'].fillna(32800)


### Find average number of dependents based on age category, education, and property type and fill null values with it
average_age = df['Age'].mean()
print(average_age)
# Filled with the average of 41
df['Age'] = df['Age'].fillna(41)

average_vehicle_age = df.groupby(['Age Category'])['Vehicle Age'].mean().reset_index()
print(average_vehicle_age)
# Filled with the average of 10
df['Vehicle Age'] = df['Vehicle Age'].fillna(10)

average_insurance_duration = df.groupby(['Age Category'])['Insurance Duration'].mean().reset_index()
print(average_insurance_duration)
# Filled with the average of 5
df['Insurance Duration'] = df['Insurance Duration'].fillna(5)

average_dependents = df.groupby(['Education Level'])['Number of Dependents'].mean().reset_index()
print(average_dependents)
# Filled with the average of 2
df['Number of Dependents'] = df['Number of Dependents'].fillna(2)

### Find average health score based on exercise frequency and smoking status and fill null values with it
average_health_score = df.groupby(['Exercise Frequency','Smoking Status'])['Health Score'].mean().reset_index()
# Filled with the average of 25.5
print(average_health_score)
df['Health Score'] = df['Health Score'].fillna(25.5)

### Find average number of claims based on age category, and fill null values with it
average_previous_claims = df.groupby(['Age Category'])['Previous Claims'].mean().reset_index()
print(average_previous_claims)
# Filled with the average of 1
df['Previous Claims'] = df['Previous Claims'].fillna(1)

### Find average credit score based on age category fill null values with it
average_credit_score = df.groupby(['Age Category'])['Credit Score'].mean().reset_index()
print(average_credit_score)
# Filled with the average of 575
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
df['Marital Status'] = df['Marital Status'].fillna('Unknown')



# List of columns to encode
columns_to_encode = [
    'Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location',
    'Policy Type', 'Customer Feedback', 'Smoking Status', 'Exercise Frequency',
    'Property Type', 'Age Category'
]

# =============================================================================
# # Apply pd.get_dummies to all specified columns, dropping the first category for each and convert to binary int
# df_encoded = df.copy()
# for col in columns_to_encode:
#     # Get the column names before encoding
#     existing_columns = set(df_encoded.columns)
#     
#     # Create dummy columns
#     df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)
# 
#     # Identify the newly created columns
#     new_columns = set(df_encoded.columns) - existing_columns
#     
#     # Convert the new columns to integers
#     df_encoded[list(new_columns)] = df_encoded[list(new_columns)].astype(int)
# =============================================================================



df_encoded = df.copy()

# Apply frequency encoding
for col in columns_to_encode:
    # Get the column names before encoding
    existing_columns = set(df_encoded.columns)
    
    freq_encoding = df_encoded[col].value_counts()  # Count occurrences
    df_encoded[f'{col}_Freq'] = df_encoded[col].map(freq_encoding)  # Map frequencies to the original data
    
    
df_encoded = df_encoded.drop(columns=columns_to_encode)

df_encoded.info()
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


## Delete out Age Category encoded fields
word = 'Age Category'
matching_columns = [col for col in df_encoded.columns if word.lower() in col.lower()]
df_encoded = df_encoded.drop(columns=matching_columns,axis=1)

df_encoded.to_csv('test_transformed.csv',index=False)














