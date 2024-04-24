import pandas as pd  
from sklearn.ensemble import IsolationForest  
from sklearn.preprocessing import StandardScaler  
  
# Load the PMU dataset into a Pandas DataFrame  
pmu_data = pd.read_csv('C:\Users\azureuser\Documents\pmu_dataset.csv')  
  
# Select the relevant features from the dataset  
features = ['Voltage_PhaseA', 'Voltage_PhaseB', 'Voltage_PhaseC',  
            'Current_PhaseA', 'Current_PhaseB', 'Current_PhaseC',  
            'Power_PhaseA', 'Power_PhaseB', 'Power_PhaseC',  
            'Frequency']  
  
# Extract the selected features from the dataset  
X = pmu_data[features].values  
  
# Standardize the feature values  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  
  
# Train the Isolation Forest model  
model = IsolationForest(contamination=0.05)  # 5% of data is considered as anomalies  
model.fit(X_scaled)  
  
# Predict the anomalies in the dataset  
predictions = model.predict(X_scaled)  
  
# Add the predictions as anomaly labels to the dataset  
pmu_data['Is_Anomaly'] = predictions  
  
# Save the updated dataset with anomaly labels  
pmu_data.to_csv('C:\Users\azureuser\Documents\pmu_dataset_with_labels.csv', index=False)  
