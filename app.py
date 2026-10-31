from datetime import datetime
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# Dropdown options (values match the dataset's categories)
AIRLINES = ["Air Asia", "Air India", "GoAir", "IndiGo", "Jet Airways",
            "Jet Airways Business", "Multiple carriers",
            "Multiple carriers Premium economy", "SpiceJet", "Trujet",
            "Vistara", "Vistara Premium economy"]
SOURCES = ["Banglore", "Chennai", "Delhi", "Kolkata", "Mumbai"]
DESTINATIONS = ["Banglore", "Cochin", "Delhi", "Hyderabad", "Kolkata", "New Delhi"]
STOPS = ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"]
INFO = ["no info", "in-flight meal not included", "no check-in baggage included",
        "1 long layover", "1 short layover", "2 long layover",
        "change airports", "business class", "red-eye flight"]


def build_features(form):
    journey = datetime.strptime(form["journey_date"], "%Y-%m-%d")
    dep = datetime.strptime(form["dep_time"], "%H:%M")
    arr = datetime.strptime(form["arr_time"], "%H:%M")

    # Duration in minutes (wraps past midnight)
    duration = int((arr - dep).total_seconds() // 60) % (24 * 60)

    # Start with every training column set to 0, in the training order
    row = pd.DataFrame(0, index=[0], columns=columns)

    numeric = {
        "Date": journey.day,
        "Month": journey.month,
        "Year": journey.year,
        "Arrival_hour": arr.hour,
        "Arrival_min": arr.minute,
        "Dep_hour": dep.hour,
        "Dep_min": dep.minute,
        "Duration_mins": duration,
    }
    for col, val in numeric.items():
        row.at[0, col] = val

    # One-hot columns: switch on the chosen one. The category dropped by
    # drop_first=True has no column, so it correctly stays all zeros.
    categorical = {
        "Airline": form["airline"],
        "Source": form["source"],
        "Destination": form["destination"],
        "Total_Stops": form["stops"],
        "Additional_Info": form["info"],
    }
    for prefix, val in categorical.items():
        col = f"{prefix}_{val}"
        if col in row.columns:
            row.at[0, col] = 1

    return row


@app.route("/", methods=["GET", "POST"])
def index():
    prediction, error = None, None

    if request.method == "POST":
        if request.form["source"] == request.form["destination"]:
            error = "Source and destination can't be the same."
        else:
            try:
                features = build_features(request.form)
                prediction = round(float(model.predict(features)[0]), 2)
            except Exception as e:
                error = f"Could not make a prediction: {e}"

    return render_template(
        "index.html",
        form=request.form, prediction=prediction, error=error,
        airlines=AIRLINES, sources=SOURCES, destinations=DESTINATIONS,
        stops=STOPS, info=INFO,
    )


if __name__ == "__main__":
    app.run(debug=True)