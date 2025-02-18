from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import os

# Dummy data
X = np.random.rand(100, 224 * 224)
y = np.random.choice(['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataracts'], 100)

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, 'models/model.pkl')
print("✅ Model saved successfully!")
