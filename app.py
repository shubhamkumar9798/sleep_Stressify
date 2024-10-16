from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Check if the model file exists
if os.path.exists('model.pkl'):
    # Try loading the model
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        print("Error loading model.pkl: ", e)
        model = None
else:
    print("model.pkl does not exist.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if model is None:
            return "Model is not available for prediction. Please retrain the model."

        # Get input from the form in predict.html
        snoring_rate = float(request.form['snoring_rate'])
        respiration_rate = float(request.form['respiration_rate'])
        body_temperature = float(request.form['body_temperature'])
        limb_movement = float(request.form['limb_movement'])
        blood_oxygen = float(request.form['blood_oxygen'])
        eye_movement = float(request.form['eye_movement'])
        sleeping_hours = float(request.form['sleeping_hours'])
        heart_rate = float(request.form['heart_rate'])

        # Create a NumPy array with the input values
        features = np.array([[snoring_rate, respiration_rate, body_temperature,
                              limb_movement, blood_oxygen, eye_movement,
                              sleeping_hours, heart_rate]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Redirect to the suggestion page based on stress level
        return render_template('suggestion.html', stress_level=prediction)

    return render_template('predict.html')

@app.route('/suggestions/<int:stress_level>')
def suggestions(stress_level):
    advice = ""
    if stress_level == 1:
        advice = "You have a low stress level. Keep up the good work!"
    elif stress_level == 2:
        advice = "Moderate stress. Try practicing meditation."
    elif stress_level == 3:
        advice = "High stress. Consider regular exercise and mindfulness."
    else:
        advice = "Very high stress. Seek professional help if needed."

    return render_template('suggestion.html', advice=advice)

if __name__ == "__main__":
    app.run(host= "0.0.0.0" ,debug=True)