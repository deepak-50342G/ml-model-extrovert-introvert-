from flask import Flask, request, render_template
import pickle
import numpy as np

# Load only the classifier
with open('extrovert_introvert_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        int_features = [int(x) for x in request.form.values()]
        final_features = np.array(int_features).reshape(1, -1)

        prediction = classifier.predict(final_features)
        output = 'Extrovert' if prediction[0] == 1 else 'Introvert'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
