from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------
# Initialize Flask
# ----------------------------------------------------------

app = Flask(__name__)

# ----------------------------------------------------------
# Load Saved Pipeline
# ----------------------------------------------------------

pipeline = joblib.load("Smart Credit Scoring.joblib")

model = pipeline["model"]
scaler = pipeline["scaler"]
feature_names = pipeline["feature_names"]
label_encoders = pipeline["label_encoders"]

print("✅ Pipeline Loaded Successfully")

# ----------------------------------------------------------
# Label Encoding Helper
# ----------------------------------------------------------

def encode(column, value):
    """
    Encode categorical values using saved LabelEncoder.
    If an unknown value is received, return 0.
    """
    if column in label_encoders:

        encoder = label_encoders[column]

        try:
            return int(encoder.transform([value])[0])

        except Exception:
            return 0

    return value

# ----------------------------------------------------------
# Create Empty Feature Dictionary
# ----------------------------------------------------------

def create_feature_dictionary():

    features = {}

    for feature in feature_names:
        features[feature] = 0

    return features

# ----------------------------------------------------------
# Calculate Engineered Features
# ----------------------------------------------------------

def calculate_engineered_features(features):

    income = features.get("AMT_INCOME_TOTAL", 1)
    credit = features.get("AMT_CREDIT", 1)
    annuity = features.get("AMT_ANNUITY", 1)
    family = max(features.get("CNT_FAM_MEMBERS", 1), 1)
    age = max(features.get("AGE_YEARS", 1), 1)
    employment = features.get("YEARS_EMPLOYED", 0)
    features["INCOME_CREDIT_RATIO"] = income / max(credit, 1)
    features["ANNUITY_INCOME_RATIO"] = annuity / max(income, 1)
    features["CREDIT_INCOME_RATIO"] = credit / max(income, 1)
    features["EMPLOYMENT_AGE_RATIO"] = employment / max(age, 1)
    features["CREDIT_PER_PERSON"] = credit / family
    features["INCOME_PER_PERSON"] = income / family
    # Convert Years -> Days
    if "DAYS_BIRTH" in features:
        features["DAYS_BIRTH"] = -(age * 365)
    if "DAYS_EMPLOYED" in features:
        features["DAYS_EMPLOYED"] = -(employment * 365)
    return features


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

        return render_template(
        "index.html",
        prediction=None,
        probability=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = create_feature_dictionary()
        gender = request.form["gender"]
        age = float(request.form["age"])
        children = int(request.form["children"])
        family = int(request.form["family"])
        income = float(request.form["income"])
        credit = float(request.form["credit"])
        annuity = float(request.form["annuity"])
        employment = float(request.form["employment"])
        contract = request.form["contract"]
        income_type = request.form["income_type"]
        education = request.form["education"]
        family_status = request.form["family_status"]
        occupation = request.form["occupation"]
        organization = request.form["organization"]
        housing = request.form["housing"]
        own_car = request.form["own_car"]
        own_house = request.form["own_house"]
        region = int(request.form["region"])
        print("✅ Form Inputs Received Successfully")
                # ==========================================================
        # FILL IMPORTANT FEATURES
        # ==========================================================

        # Personal Details
        features["CODE_GENDER"] = encode("CODE_GENDER", gender)
        features["CNT_CHILDREN"] = children
        features["CNT_FAM_MEMBERS"] = family
        # Financial Details
        features["AMT_INCOME_TOTAL"] = income
        features["AMT_CREDIT"] = credit
        features["AMT_ANNUITY"] = annuity
        # Employment
        features["YEARS_EMPLOYED"] = employment
        features["AGE_YEARS"] = age
        # Loan Details
        features["NAME_CONTRACT_TYPE"] = encode(
            "NAME_CONTRACT_TYPE",
            contract
        )
        features["NAME_INCOME_TYPE"] = encode(
            "NAME_INCOME_TYPE",
            income_type
        )
        features["NAME_EDUCATION_TYPE"] = encode(
            "NAME_EDUCATION_TYPE",
            education
        )
        features["NAME_FAMILY_STATUS"] = encode(
            "NAME_FAMILY_STATUS",
            family_status
        )
        features["OCCUPATION_TYPE"] = encode(
            "OCCUPATION_TYPE",
            occupation
        )
        features["ORGANIZATION_TYPE"] = encode(
            "ORGANIZATION_TYPE",
            organization
        )
        features["NAME_HOUSING_TYPE"] = encode(
            "NAME_HOUSING_TYPE",
            housing
        )
        features["FLAG_OWN_CAR"] = encode(
            "FLAG_OWN_CAR",
            own_car
        )
        features["FLAG_OWN_REALTY"] = encode(
            "FLAG_OWN_REALTY",
            own_house
        )
        features["REGION_RATING_CLIENT"] = region
        # ==========================================================
        # CALCULATE ENGINEERED FEATURES
        # ==========================================================

        features = calculate_engineered_features(features)
        # ==========================================================
        # CREATE DATAFRAME
        # ==========================================================
        input_df = pd.DataFrame(
            [features],
            columns=feature_names
        )
        # ==========================================================
        # APPLY STANDARD SCALER
        # ==========================================================
        try:

            input_scaled = scaler.transform(input_df)
            print("\n========== DEBUG ==========")
            print("Input Data:")
            print(input_df.T)
            print("\nScaled Shape:", input_scaled.shape)
            print("\nPrediction:", model.predict(input_scaled))
            print("Probability:", model.predict_proba(input_scaled))
            print("===========================\n")
        except Exception:

            input_scaled = input_df
        print("Prediction:", model.predict(input_scaled))
        print("Probability:", model.predict_proba(input_scaled))
        print(input_df.T)
        prediction = model.predict(input_scaled)[0]
        probability = None
        try:

            probability = model.predict_proba(
                input_scaled
            )[0][1]
        except:
            pass
                # ==========================================================
        # PREPARE RESULT
        # ==========================================================
        if prediction == 0:
            result = "✅ Eligible for Loan"
        else:
            result = "❌ High Default Risk"
        return render_template(
            "index.html",
            prediction=result,
            probability=probability
        )
    # ==========================================================
    # ERROR HANDLING
    # ==========================================================

    except Exception as e:
        print("Prediction Error :", e)
        return render_template(
            "index.html",
            prediction=f"Error : {str(e)}",
            probability=None
        )
# ==========================================================
# RUN FLASK
# ==========================================================
if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )