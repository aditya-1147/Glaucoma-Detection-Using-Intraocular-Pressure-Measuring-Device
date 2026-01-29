# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.utils import shuffle

"""## Data Exploration"""

df = pd.read_csv('glaucoma_dataset.csv')

df.head()

df.info()

#Checking the sum of diagnosed Glaucoma
num_of_diagnosed= (df['Diagnosis'] == 'Glaucoma').sum()
num_of_diagnosed

cols = ['Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)']
df[cols].describe()

"""- Youngest patient is 18 years old and the oldest is 90 years old.
- Mean age of the patients is 54 years old.
"""

df.hist(column='Cup-to-Disc Ratio (CDR)', by='Diagnosis')

"""- The above graph shows that a large cup-to-disc ratio doesn’t necessarily mean that individuals are diagnosed with glaucoma"""

sns.scatterplot(x = 'Age', y = 'Gender', hue = 'Diagnosis', data = df)
plt.title('Relationship between Gender, Age and Glaucoma');

"""- The distribution of the graph above shows that females tend to have a higher risk of diagnosed glaucoma as their age increases."""

sns.scatterplot(x = 'Intraocular Pressure (IOP)', y = 'Cup-to-Disc Ratio (CDR)', hue = 'Diagnosis', data = df)
plt.title('Relationship between Cup-to-Disc Ratio, Intraocular Pressure and Glaucoma');

GlaucomaType_count = df['Glaucoma Type'].value_counts()
GlaucomaType_count

sns.barplot(x=GlaucomaType_count.values, y=GlaucomaType_count.index, orient='h')

plt.title('Glaucoma Type Distribution')
plt.xlabel('Count')
plt.ylabel('Glaucoma Type')
plt.show()

"""## Data Cleaning & Preprocessing"""

#Drop irrelevant data
df.drop(columns=['Patient ID','Glaucoma Type'], inplace=True)

#Check what's in VA Measurements
unique_VA_values = df['Visual Acuity Measurements'].unique()
unique_VA_values

#Standardise VA Measurements format to only LogMAR

df['LogMAR VA'] = None

for index, row in df.iterrows():
    va_measurement = row['Visual Acuity Measurements']
    if 'LogMAR' in va_measurement:
        logmar_value = float(va_measurement.split()[1])
    elif va_measurement == '20/40':
        logmar_value = 0.3
    elif va_measurement == '20/20':
        logmar_value = 0.0
    else:
        logmar_value = None
    df.at[index, 'LogMAR VA'] = logmar_value

df[['Visual Acuity Measurements', 'LogMAR VA']]

#Drop 'Visual Acuity Measurements'
df.drop(columns=['Visual Acuity Measurements'], inplace=True)

#Check unique values in VFT Results
unique_VFT_values = df['Visual Field Test Results'].unique()
unique_VFT_values

#Add all the VFT results to their own column
df['VFT Sensitivity'] = None
df['VFT Specificity'] = None

for index, row in df.iterrows():
    vft_results = row['Visual Field Test Results']
    values = vft_results.split(', ')
    if len(values) == 2:
        sensitivity = float(values[0].split(': ')[1])
        specificity = float(values[1].split(': ')[1])
        df.at[index, 'VFT Sensitivity'] = sensitivity
        df.at[index, 'VFT Specificity'] = specificity

df[['VFT Sensitivity', 'VFT Specificity']]

#Drop Visual Field Test Results
df.drop(columns=['Visual Field Test Results'], inplace=True)

#Check unique values in OCT Results
unique_OCT_values = df['Optical Coherence Tomography (OCT) Results'].unique()
unique_OCT_values

#Add all the OCT results to their own column

df['OCT RNFL Thickness (µm)'] = None
df['OCT GCC Thickness (µm)'] = None
df['OCT Retinal Volume (mm³)'] = None
df['OCT Macular Thickness (µm)'] = None

for index, row in df.iterrows():
    oct_results = row['Optical Coherence Tomography (OCT) Results']
    values = re.findall(r'\d+\.\d+', oct_results)

    if len(values) == 4:
        df.at[index, 'OCT RNFL Thickness (µm)'] = float(values[0])
        df.at[index, 'OCT GCC Thickness (µm)'] = float(values[1])
        df.at[index, 'OCT Retinal Volume (mm³)'] = float(values[2])
        df.at[index, 'OCT Macular Thickness (µm)'] = float(values[3])

df[['OCT RNFL Thickness (µm)', 'OCT GCC Thickness (µm)', 'OCT Retinal Volume (mm³)', 'OCT Macular Thickness (µm)']]

#Drop Optical Coherence Tomography (OCT) Results
df.drop(columns=['Optical Coherence Tomography (OCT) Results'], inplace=True)

#convert VFT results and OCT results's dtype into float instead of object
columns_to_convert = [
    'VFT Sensitivity',
    'VFT Specificity',
    'OCT RNFL Thickness (µm)',
    'OCT GCC Thickness (µm)',
    'OCT Retinal Volume (mm³)',
    'OCT Macular Thickness (µm)'
]

for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#Check unique value
df['Medication Usage'].unique()

#Fill up all the null values for ease data process
df.fillna(value=pd.NaT, inplace=True)

#Making all the meds combination to their own column
unique_medications = set()

for meds in df['Medication Usage']:
    if pd.notna(meds):                  # Check if the value is not NaN
        meds_list = meds.split(', ')    # Split the meds
        unique_medications.update(meds_list)

for med in unique_medications:          # Add 1/0 to all the meds accordingly
    df[med] = df['Medication Usage'].apply(lambda x: 1 if (pd.notna(x) and med in x) else 0)

# Drop the original "Medication Usage" column
df.drop(columns=['Medication Usage'], inplace=True)

df.head()

#Repeat the same process for 'Visual Symptoms'
unique_symptoms = set()

for symptoms in df['Visual Symptoms']:
    symptoms_list = symptoms.split(', ')
    unique_symptoms.update(symptoms_list)

for symptom in unique_symptoms:
    df[symptom] = df['Visual Symptoms'].apply(lambda x: 1 if symptom in x else 0)

# Drop the original "Visual Symptoms" column
df.drop(columns=['Visual Symptoms'], inplace=True)

df.head()

#Map Diagnosis to 1/0
df['Diagnosis'] = df['Diagnosis'].map({'No Glaucoma':0,'Glaucoma':1})

#Dummify the rest of the columns
df = pd.get_dummies(data=df, columns=['Gender', 'Family History', 'Medical History','Cataract Status', 'Angle Closure Status'], dtype=int)

#Oversampling the data
df_majority = df[df['Diagnosis'] == 0]
df_minority = df[df['Diagnosis'] == 1]

oversampling_factor = 4

df_minority_oversampled = df_minority.sample(n=len(df_majority) * oversampling_factor, replace=True)

df_oversampled = pd.concat([df_majority, df_minority_oversampled])

df_oversampled = shuffle(df_oversampled, random_state=42)

df_oversampled['Diagnosis'].value_counts()

df_oversampled.info()

"""## Modelling"""

# Split the data into features (X) and target (y)
X = df_oversampled.drop(['Diagnosis'], axis=1)
y = df_oversampled['Diagnosis']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .10, stratify=y)

y.value_counts(normalize = False)

#Defining hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

#Initialize the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

#Perform Gridsearch
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=16)
grid_search.fit(X_train, y_train)

#Get the best parameters
best_params = grid_search.best_params_
best_params

#Train the model with the best parameters
best_rf_model = RandomForestClassifier(**best_params, random_state=42)
best_rf_model.fit(X_train, y_train)

#Prediction
y_pred = best_rf_model.predict(X_test)

#Crss Validation Score
cross_val_score(best_rf_model, X_train, y_train, cv=3).mean()

#Evaluation
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report_str)

# Save the trained model
import pickle

print("\n💾 Saving trained model...")
with open('glaucoma_model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)
print("✅ Model saved as 'glaucoma_model.pkl'")

# Save feature names for later use
feature_names = X.columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✅ Feature names saved as 'feature_names.pkl'")

