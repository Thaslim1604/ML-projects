import joblib
import hashlib

pipeline = joblib.load("Smart Credit Scoring.joblib")

print(type(pipeline["model"]))
print(pipeline["feature_names"][:20])

with open("Smart Credit Scoring.joblib", "rb") as f:
    print(hashlib.md5(f.read()).hexdigest())