from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
standard_scaler = StandardScaler()

def preprocess_input(Year, Present_Price, Kms_Driven, Owner, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Mannual):
    # Input validation and preprocessing
    try:
        Year = int(Year)
        Present_Price = float(Present_Price)
        Kms_Driven = int(Kms_Driven)
        Kms_Driven2 = np.log(Kms_Driven)
        Owner = int(Owner)
        Fuel_Type_Diesel = 1 if Fuel_Type_Petrol == 'Diesel' else 0
        Seller_Type_Individual = 1 if Seller_Type_Individual == 'Individual' else 0
        Transmission_Mannual = 1 if Transmission_Mannual == 'Mannual' else 0
        Year = 2020 - Year

        return [Present_Price, Kms_Driven2, Owner, Year, Fuel_Type_Diesel, 1 - Fuel_Type_Diesel,
                Seller_Type_Individual, Transmission_Mannual]
    except ValueError:
        return None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = preprocess_input(request.form['Year'], request.form['Present_Price'],
                                      request.form['Kms_Driven'], request.form['Owner'],
                                      request.form['Fuel_Type_Petrol'], request.form['Seller_Type_Individual'],
                                      request.form['Transmission_Mannual'])

        if input_data is not None:
            prediction = model.predict([input_data])
            output = round(prediction[0], 2)

            if output < 0:
                return render_template('index.html', prediction_text="Sorry, you cannot sell this car")
            else:
                return render_template('index.html', prediction_text="You can sell the car at {}".format(output))
        else:
            return render_template('index.html', prediction_text="Invalid input. Please check your input values.")
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)