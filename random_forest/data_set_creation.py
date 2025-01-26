import numpy as np
import pandas as pd

# Generate synthetic data (by repeating the symptoms with slight variations)
num_samples = 100  # Increase the number of samples
symptoms = ['Headache', 'Chest pain', 'Cough', 'Nausea', 'Fever', 'Dizziness', 'Fatigue', 'Back pain', 'Joint pain',
            'Shortness of breath', 'Cold', 'Loss of appetite', 'Body ache', 'Loss of taste', 'Sore throat']

doctor_types = ['General Physician', 'Cardiologist', 'ENT Specialist', 'GP', 'Neurologist', 'Orthopedist', 'Pulmonologist']

# Generate random symptoms and doctor types
generated_data = {
    'Symptom 1': np.random.choice(symptoms, num_samples),
    'Symptom 2': np.random.choice(symptoms, num_samples),
    'Symptom 3': np.random.choice(symptoms, num_samples),
    'Doctor Type': np.random.choice(doctor_types, num_samples)
}

# Print the generated data before creating the DataFrame
print(generated_data)

# Create a DataFrame from the generated data
df_synthetic = pd.DataFrame(generated_data)

# Show the first few rows of the DataFrame
print(df_synthetic.head())
