from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and label encoders
model = joblib.load('model.pkl')  # Make sure this file exists
le_gender = joblib.load('le_gender.pkl')  # Load gender label encoder
le_family = joblib.load('le_family.pkl')  # Load family history label encoder

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get form data
            age = int(request.form['age'])
            gender = request.form['gender']
            family_history = request.form['family_history']

            # Encode categorical variables (make sure you have the label encoders loaded)
            gender_encoded = le_gender.transform([gender])[0]
            family_history_encoded = le_family.transform([family_history])[0]

            # Make prediction
            prediction_value = model.predict(np.array([[age, gender_encoded, family_history_encoded]]))[0]

            # Decode prediction to a readable result
            if prediction_value == 0:
                prediction = "No Treatment Needed"
            else:
                prediction = "Treatment Required"
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
