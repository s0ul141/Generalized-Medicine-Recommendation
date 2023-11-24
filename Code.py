#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
data = pd.read_csv("C:/Users/soumi/Downloads/med_data.csv")
len(data)


# In[34]:


# Data preprocessing
data.dropna(subset=['Height','Age','Weight','Gender','Blood Group','Blood Pressure'], inplace=True)
data = data[data['Age'].astype(str).str.isnumeric()]
data = data[data['Weight'] > 0]

# Display basic statistics
print("Basic Statistics:")
print(data.describe())
print("\n")


# In[35]:


# Calculate BMI
data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)
# Define BMI categories
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# Apply the categorization function to the 'BMI' column
data['BMI Category'] = data['BMI'].apply(categorize_bmi)
data


# In[36]:


#Save the DataFrame as a CSV file
data.to_csv('C:/Users/soumi/Downloads/new_med.csv', index=False)


# In[37]:


# Calculate average weight and BMI
average_weight = data['Weight'].mean()
average_bmi = data['BMI'].mean()

# Display average weight and BMI
print(f"Average Weight: {average_weight:.2f} kgs")
print(f"Average BMI: {average_bmi:.2f}") 
print("\n")

# Plot BMI Distribution by Category
bmi_category_counts = data['BMI Category'].value_counts()
print("BMI Distribution by Category:")
print(bmi_category_counts)
# Specify the order of categories for plotting
bmi_category_order = ["Underweight", "Normal", "Overweight", "Obese"]

bmi_category_counts.loc[bmi_category_order].plot(kind='bar', color='green')
plt.title("BMI Distribution by Category")
plt.xlabel("BMI Category")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
print("\n")


# In[38]:


# Define height categories
def categorize_height(height_cm):
    if height_cm <= 149:
        return "Very Short"
    elif 150 <= height_cm <= 159:
        return "Short"
    elif 160 <= height_cm <= 169:
        return "Medium"
    elif 170 <= height_cm <=179:
        return "Tall"
    else:
        return "Very Tall"

# Apply the categorization function to the 'Height (in cm)' column
data['Height Category'] = data['Height'].apply(categorize_height)

# Plot Height Distribution by Category
height_category_counts = data['Height Category'].value_counts()
print("Height Distribution by Category:")
print(height_category_counts)
# Specify the order of categories for plotting
category_order = ["Very Short", "Short", "Medium", "Tall", "Very Tall"]

height_category_counts.loc[category_order].plot(kind='bar', color='blue')
plt.title("Height Distribution by Category")
plt.xlabel("Height Category")
plt.ylabel("Count")
plt.xticks(rotation=0)
# Add height limits annotations to the plot
height_limits = {
    "Very Short": "<= 149 cm",
    "Short": "150 - 159 cm",
    "Medium": "160 - 169 cm",
    "Tall": "170 - 179 cm",
    "Very Tall": ">= 180 cm"
}

for idx, category in enumerate(category_order):
    plt.text(idx, height_category_counts[category] + 10, height_limits[category], ha='center')

average_height= data['Height'].mean()
print(f"Average height is: {average_height:.2f} cm")
plt.tight_layout()
plt.show()
print("\n")
# Plot height distribution
data['Height'].plot(kind='hist', bins=20, color='blue', edgecolor='black')
plt.axvline(average_height, color='red', linestyle='dashed', linewidth=2, label='Avg Height')
plt.title("Height Distribution")
plt.xlabel("Height (in cm)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()
print("\n")


# In[39]:


# Plot gender distribution
gender_counts = data['Gender'].value_counts()
print("Gender Distribution:")
print(gender_counts)
gender_counts.plot(kind='bar', color='skyblue')
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
print("\n")


# In[40]:


# Plot blood group distribution
blood_group_counts = data['Blood Group'].value_counts()
print("Blood Group Distribution:")
print(blood_group_counts)
blood_group_counts.plot(kind='bar', color='lightgreen')
plt.title("Blood Group Distribution")
plt.xlabel("Blood Group")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
print("\n")


# In[42]:


# Most common allergies
allergies = data['Allergy You Have'].str.split(', ', expand=True).stack().value_counts()
print("Most Common Allergies:")
print(allergies)
allergies[:10].plot(kind='barh', color='orange')
plt.title("Most Common Allergies")
plt.xlabel("Count")
plt.ylabel("Allergy")
plt.tight_layout()
plt.show()
print("\n")


# In[44]:


# Average age of participants
average_age = data['Age'].mean()
print(f"Average Age: {average_age:.2f}")
print("\n")

# Plot age distribution
data['Age'].plot(kind='hist', bins=20, color='purple', edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
print("\n")


# In[46]:


# Common symptoms or health issues
symptoms = data['What are the current symptoms or health issues you are facing'].str.split(', ', expand=True).stack().value_counts()
print("Common Symptoms or Health Issues:")
print(symptoms)
symptoms[:10].plot(kind='barh', color='pink')
plt.title("Common Symptoms or Health Issues")
plt.xlabel("Count")
plt.ylabel("Symptom")
plt.tight_layout()
plt.show()
print("\n")


# # Model Building

# In[47]:


import pandas as pd

# Load the dataset
df = pd.read_csv("C:/Users/soumi/Downloads/med_data.csv")
df


# In[48]:


#Preprocess data
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
df['Medical Health History'] = LabelEncoder().fit_transform(df['Medical Health History'])
df['Allergy You Have'] = LabelEncoder().fit_transform(df['Allergy You Have'])
df


# In[50]:


from io import StringIO

# Select only the 'Medication you take...' columns
medication_columns = df.iloc[:, 12:-1]

# Combine the values from each row and store in a new column
df['All Medications'] = medication_columns.apply(lambda row: ', '.join(row.dropna()), axis=1)

# Display the updated DataFrame
df.head()


# In[60]:


# Select only the 'Name' column and the 'Medication you take...' columns
medication_columns = df.iloc[:, 2:-1]

# Create a list of conditions and corresponding medications
conditions_and_meds = []
for index, row in medication_columns.iterrows():
    for condition, medication in row.items():
        if condition.startswith('Medication you take') and medication:
            conditions_and_meds.append([condition.replace('Medication you take for ', ''), medication])

# Create a new DataFrame for conditions and medications
conditions_df = pd.DataFrame(conditions_and_meds, columns=['Condition', 'Medication'])

conditions_df


# In[62]:


# Drop rows with missing values in the 'Medication' column
conditions_df = conditions_df.dropna(subset=['Medication'])
conditions_df = conditions_df[conditions_df['Medication'].astype(str).str.isalpha()]
conditions_df


# In[63]:


# List of values to drop
values_to_drop = ["-","None", "Nil", "Nil ", "no ", "Nothing ", "No", "Yes", "Nothing", "Na", "Yes9", "Some time", "NIL", "None ", "Y","N", "no ", "No ", "Yes ", "Nothing ", "Na ", "Yes9 ", "Some time ", "NIL "]

# Drop rows with specified values in the 'Medication' column
conditions_df = conditions_df[~conditions_df['Medication'].isin(values_to_drop)]

conditions_df


# In[65]:


# Save the conditions and medications DataFrame as a CSV file
conditions_df.to_csv('C:/Users/soumi/Downloads/conditions_and_medications.csv', index=False)

print("CSV file saved successfully.")
conditions_df


# # Model

# In[70]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the preprocessed dataset (conditions_df)
conditions_df = pd.read_csv('C:/Users/soumi/Downloads/conditions_and_medications.csv')

# Create a dictionary to map conditions to labels
condition_to_label = {condition: label for label, condition in enumerate(conditions_df['Condition'].unique())}

# Add a new column with encoded conditions in the dataset
conditions_df['Condition_Label'] = conditions_df['Condition'].map(condition_to_label)

# Create and train a Decision Tree Classifier
model = DecisionTreeClassifier()
X_train = conditions_df[['Condition_Label']]
y_train = conditions_df['Medication']
model.fit(X_train, y_train)

# Make predictions on the training data
y_pred = model.predict(X_train)

# Calculate and print the accuracy on the training data
accuracy = accuracy_score(y_train, y_pred)
print("Accuracy on Training Data:", accuracy)


# In[71]:


# List of available conditions
available_conditions = conditions_df['Condition'].unique()

# Get symptoms input from the user
entered_symptoms = []
print("Available conditions to choose from:")
for idx, condition in enumerate(available_conditions):
    print(f"{idx + 1}. {condition}")
while True:
    choice = input("Enter a number corresponding to a symptom (or 'done' to finish): ")
    if choice.lower() == 'done':
        break
    try:
        index = int(choice) - 1
        if 0 <= index < len(available_conditions):
            entered_symptoms.append(available_conditions[index])
        else:
            print("Invalid choice. Please enter a valid number.")
    except ValueError:
        print("Invalid input. Please enter a number or 'done'.")

# Encode entered symptoms using the condition-to-label mapping
entered_symptoms_encoded = [condition_to_label[symptom] for symptom in entered_symptoms]

# Predict medications for the entered symptoms
predicted_medications = model.predict(pd.DataFrame(entered_symptoms_encoded, columns=['Condition_Label']))

# Print the predicted medications
print("Predicted Medications:")
for symptom, medication in zip(entered_symptoms, predicted_medications):
    print(f"For symptom '{symptom}', predicted medication: {medication}")


# In[72]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder



# Create a Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Medication Predictor"),
    html.Label("Select Symptoms:"),
    dcc.Dropdown(
        id='symptoms-dropdown',
        options=[{'label': condition, 'value': condition} for condition in available_conditions],
        multi=True
    ),
    html.Div(id='output-container')
])

# Define callback to update output based on selected symptoms
@app.callback(
    Output('output-container', 'children'),
    Input('symptoms-dropdown', 'value')
)
def update_output(selected_symptoms):
    if selected_symptoms:
        # Encode selected symptoms using the condition-to-label mapping
        selected_symptoms_encoded = [condition_to_label[symptom] for symptom in selected_symptoms]

        # Predict medications for the selected symptoms
        predicted_medications = model.predict(pd.DataFrame(selected_symptoms_encoded, columns=['Condition_Label']))

        # Create a list of predicted medications
        medications_list = []
        for symptom, medication in zip(selected_symptoms, predicted_medications):
            medications_list.append(f"For symptom '{symptom}', predicted medication: {medication}")

        return html.Div([html.P(medication) for medication in medications_list])
    else:
        return ""

if __name__ == '__main__':
    app.run_server(debug=True)

