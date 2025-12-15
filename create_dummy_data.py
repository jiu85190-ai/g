import pandas as pd
import numpy as np

# Create dummy data for Crop_recommendation.csv
np.random.seed(42)
n_samples = 1000
crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']

data = {
    'temperature': np.random.uniform(5, 45, n_samples),
    'label': np.random.choice(crops, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('Crop_recommendation.csv', index=False)
print("Dummy csv created")
