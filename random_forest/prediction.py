import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Corrected dataset with equal-length lists
data = {
    'Symptom 1': ['Fever', 'Chest pain', 'Sore throat', 'Dizziness', 'Chest pain',
       'Headache', 'Headache', 'Fever', 'Cough', 'Body ache',
       'Loss of appetite', 'Cold', 'Joint pain', 'Shortness of breath',
       'Fever', 'Loss of taste', 'Back pain', 'Sore throat', 'Body ache',
       'Chest pain', 'Cold', 'Loss of appetite', 'Dizziness', 'Dizziness',
       'Cough', 'Loss of appetite', 'Chest pain', 'Nausea', 'Cold',
       'Sore throat', 'Nausea', 'Sore throat', 'Joint pain', 'Back pain',
       'Back pain', 'Back pain', 'Back pain', 'Cough', 'Loss of appetite',
       'Fever', 'Fatigue', 'Loss of appetite', 'Cold', 'Joint pain',
       'Back pain', 'Headache', 'Cold', 'Chest pain', 'Fever',
       'Sore throat', 'Chest pain', 'Cough', 'Sore throat', 'Chest pain',
       'Fever', 'Sore throat', 'Cough', 'Shortness of breath', 'Cold',
       'Sore throat', 'Headache', 'Sore throat', 'Dizziness', 'Nausea',
       'Sore throat', 'Loss of taste', 'Chest pain', 'Cold', 'Fever',
       'Shortness of breath', 'Cold', 'Body ache', 'Back pain',
       'Headache', 'Loss of taste', 'Dizziness', 'Shortness of breath',
       'Fever', 'Cough', 'Cold', 'Fever', 'Loss of appetite',
       'Joint pain', 'Cold', 'Chest pain', 'Loss of taste', 'Cold',
       'Cold', 'Loss of taste', 'Chest pain', 'Loss of appetite',
       'Chest pain', 'Fatigue', 'Nausea', 'Sore throat', 'Sore throat',
       'Dizziness', 'Nausea', 'Back pain', 'Fever'],
    'Symptom 2': ['Back pain', 'Sore throat', 'Sore throat', 'Shortness of breath',
       'Shortness of breath', 'Loss of appetite', 'Cough', 'Headache',
       'Sore throat', 'Cold', 'Cough', 'Cough', 'Dizziness',
       'Shortness of breath', 'Shortness of breath', 'Cough', 'Body ache',
       'Back pain', 'Joint pain', 'Headache', 'Fatigue', 'Fever',
       'Joint pain', 'Cough', 'Dizziness', 'Loss of taste', 'Fatigue',
       'Fever', 'Cold', 'Dizziness', 'Sore throat', 'Dizziness', 'Cough',
       'Cough', 'Sore throat', 'Dizziness', 'Loss of appetite', 'Cold',
       'Loss of appetite', 'Back pain', 'Cold', 'Dizziness', 'Headache',
       'Sore throat', 'Headache', 'Joint pain', 'Cold', 'Fatigue',
       'Fatigue', 'Sore throat', 'Back pain', 'Dizziness', 'Fever',
       'Fever', 'Back pain', 'Body ache', 'Loss of taste', 'Dizziness',
       'Back pain', 'Cough', 'Body ache', 'Fatigue', 'Back pain',
       'Loss of taste', 'Dizziness', 'Fatigue', 'Loss of taste',
       'Sore throat', 'Headache', 'Shortness of breath', 'Joint pain',
       'Cold', 'Loss of taste', 'Back pain', 'Joint pain', 'Cold',
       'Nausea', 'Body ache', 'Joint pain', 'Dizziness',
       'Shortness of breath', 'Joint pain', 'Cough', 'Headache', 'Cold',
       'Headache', 'Cold', 'Body ache', 'Joint pain', 'Headache',
       'Body ache', 'Body ache', 'Body ache', 'Joint pain', 'Nausea',
       'Shortness of breath', 'Body ache', 'Cold', 'Dizziness',
       'Chest pain'],
    'Symptom 3': ['Shortness of breath', 'Headache', 'Sore throat', 'Cough', 'Cough',
       'Back pain', 'Headache', 'Chest pain', 'Fatigue', 'Loss of taste',
       'Chest pain', 'Joint pain', 'Loss of taste', 'Loss of taste',
       'Dizziness', 'Dizziness', 'Nausea', 'Body ache',
       'Loss of appetite', 'Sore throat', 'Nausea', 'Dizziness', 'Fever',
       'Loss of appetite', 'Joint pain', 'Shortness of breath',
       'Shortness of breath', 'Fever', 'Chest pain', 'Joint pain',
       'Joint pain', 'Cold', 'Back pain', 'Chest pain', 'Fatigue',
       'Body ache', 'Back pain', 'Fatigue', 'Loss of taste', 'Fever',
       'Fever', 'Shortness of breath', 'Nausea', 'Body ache',
       'Joint pain', 'Loss of taste', 'Back pain', 'Fever', 'Fatigue',
       'Headache', 'Fever', 'Loss of taste', 'Body ache', 'Fatigue',
       'Fatigue', 'Dizziness', 'Back pain', 'Loss of appetite',
       'Chest pain', 'Nausea', 'Joint pain', 'Fever', 'Back pain',
       'Cough', 'Back pain', 'Sore throat', 'Fatigue', 'Fatigue', 'Fever',
       'Dizziness', 'Body ache', 'Cough', 'Back pain',
       'Shortness of breath', 'Loss of taste', 'Chest pain', 'Joint pain',
       'Loss of taste', 'Chest pain', 'Cold', 'Headache', 'Cold',
       'Dizziness', 'Joint pain', 'Chest pain', 'Cough', 'Cold',
       'Chest pain', 'Fatigue', 'Cough', 'Loss of appetite', 'Joint pain',
       'Loss of appetite', 'Cough', 'Shortness of breath',
       'Shortness of breath', 'Nausea', 'Cough', 'Fatigue', 'Back pain'],
    'Doctor Type': ['Cardiologist', 'General Physician', 'ENT Specialist',
       'ENT Specialist', 'Neurologist', 'Pulmonologist', 'Pulmonologist',
       'General Physician', 'Orthopedist', 'Orthopedist', 'Neurologist',
       'Cardiologist', 'Pulmonologist', 'General Physician', 'GP',
       'Pulmonologist', 'Cardiologist', 'Neurologist', 'Cardiologist',
       'General Physician', 'Pulmonologist', 'General Physician',
       'General Physician', 'General Physician', 'General Physician',
       'Cardiologist', 'Cardiologist', 'Orthopedist', 'Neurologist',
       'Neurologist', 'Orthopedist', 'General Physician',
       'General Physician', 'ENT Specialist', 'Cardiologist',
       'Orthopedist', 'Pulmonologist', 'Orthopedist', 'General Physician',
       'General Physician', 'Pulmonologist', 'GP', 'General Physician',
       'Cardiologist', 'ENT Specialist', 'ENT Specialist', 'Neurologist',
       'General Physician', 'General Physician', 'Pulmonologist',
       'General Physician', 'General Physician', 'GP', 'Pulmonologist',
       'Cardiologist', 'GP', 'Cardiologist', 'GP', 'GP', 'Orthopedist',
       'General Physician', 'General Physician', 'ENT Specialist',
       'Neurologist', 'Orthopedist', 'Pulmonologist', 'Orthopedist',
       'ENT Specialist', 'GP', 'GP', 'Orthopedist', 'Cardiologist',
       'Cardiologist', 'Neurologist', 'Pulmonologist', 'Cardiologist',
       'ENT Specialist', 'General Physician', 'Cardiologist',
       'Neurologist', 'Pulmonologist', 'Neurologist', 'GP',
       'General Physician', 'Cardiologist', 'General Physician',
       'Orthopedist', 'Cardiologist', 'ENT Specialist',
       'General Physician', 'ENT Specialist', 'ENT Specialist',
       'Neurologist', 'Cardiologist', 'Cardiologist', 'GP',
       'ENT Specialist', 'ENT Specialist', 'Orthopedist', 'Cardiologist'],
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Create a separate LabelEncoder for symptoms and doctor type
symptom_encoder = LabelEncoder()
doctor_encoder = LabelEncoder()

# Fit the encoder for each symptom column
df['Symptom 1'] = symptom_encoder.fit_transform(df['Symptom 1'])
df['Symptom 2'] = symptom_encoder.fit_transform(df['Symptom 2'])
df['Symptom 3'] = symptom_encoder.fit_transform(df['Symptom 3'])

# Fit the doctor encoder for the 'Doctor Type' column
df['Doctor Type'] = doctor_encoder.fit_transform(df['Doctor Type'])

# Split the data into features (X) and labels (y)
X = df[['Symptom 1', 'Symptom 2', 'Symptom 3']]
y = df['Doctor Type']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model with the training data
random_forest.fit(X_train, y_train)

# Make predictions on the test data
y_pred = random_forest.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Debugging: Print out X_train, y_train, and y_pred to diagnose the issue
print("X_train:", X_train)
print("y_train:", y_train)
print("y_pred:", y_pred)

# Now use the symptom encoder and doctor encoder for prediction
new_symptoms = pd.DataFrame({
    'Symptom 1': [symptom_encoder.transform(['Headache'])[0]],  # Transforming the input symptom to a numeric value
    'Symptom 2': [symptom_encoder.transform(['Fever'])[0]],
    'Symptom 3': [symptom_encoder.transform(['Fatigue'])[0]]
})

# Predict the doctor type for the new symptoms
predicted_doctor_type = random_forest.predict(new_symptoms)

# Decode the predicted doctor type back to the original label
predicted_doctor = doctor_encoder.inverse_transform(predicted_doctor_type)
print(f"Predicted Doctor: {predicted_doctor[0]}")
