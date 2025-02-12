from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

# Load Iris dataset
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode target labels
encoder = LabelBinarizer()
y_encoded = encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Create ANN model
model = Sequential([
    Dense(8, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=0)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('D:/WebiSoftTech/ANN/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        data = [float(request.form.get(key)) for key in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        features = np.array([data])
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)
        predicted_class = encoder.inverse_transform(prediction)

        return jsonify({"Prediction": iris_data.target_names[predicted_class[0]]})

    except Exception as e:
        return jsonify({"Error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)