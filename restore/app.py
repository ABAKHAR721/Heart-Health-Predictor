from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__, template_folder='Flask_heart/templates')

# Load the saved model and scaler
with open('./Flask_heart/model/logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('./Flask_heart/model/scaler_logistic_regression_model.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',percentage=None,prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.form.to_dict()
    
    # Convert data to DataFrame
    df = pd.DataFrame([data])
    
    # Scale the features
    df_scaled = scaler.transform(df)

    # Make predictions
    prediction = model.predict(df_scaled)
    
    # Predict probabilities
    probabilities = model.predict_proba(df_scaled)[:, 1]  # Probability of class 1

    # Convert probability to percentage
    percentage = probabilities[0] * 100

    # Render the template with the prediction and percentage
    return render_template('index.html', 
                           probabilities=probabilities[0],
                           prediction=int(prediction[0]), 
                           percentage=f"{percentage:.2f}")
if __name__ == '__main__':
    app.run(debug=True)
