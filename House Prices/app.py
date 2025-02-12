# Import necessary libraries
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv('D:/WebiSoftTech/ANN/House Prices/housepricedata.csv')

# Preprocess the data
X = data.drop('AboveMedianPrice', axis=1)
y = data['AboveMedianPrice']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build the ANN model
model = Sequential()
model.add(Dense(units=32, activation='relu', input_shape=(X_scaled.shape[1],)))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_scaled, y, epochs=50, batch_size=10)

# Create Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json(force=True)
    features = pd.DataFrame(data, index=[0])
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    result = (prediction[0][0] > 0.5).astype(int)  # Convert to binary
    
    return jsonify({'AboveAverage': bool(result)})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)