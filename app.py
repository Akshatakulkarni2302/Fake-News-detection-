from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    # Basic preprocessing (optional)
    text_proc = text.strip()

    # Transform
    try:
        vect = vectorizer.transform([text_proc])
    except Exception as e:
        print("Vectorizer transform error:", e)
        return render_template('index.html', prediction="Vectorizer error")

    # Model prediction
    try:
        pred = model.predict(vect)
    except Exception as e:
        print("Model predict error:", e)
        return render_template('index.html', prediction="Model error")

    # If model has predict_proba, show probabilities
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vect)
    except Exception as e:
        print("predict_proba error:", e)

    # Debug prints to terminal
    print("=== DEBUG PREDICTION ===")
    print("Input text:", text_proc)
    print("Vector shape:", getattr(vect, "shape", None))
    print("Model type:", type(model))
    print("Prediction raw:", pred)
    if proba is not None:
        print("Predict_proba:", proba)
    # If classifier has classes_ attribute, print it
    if hasattr(model, "classes_"):
        print("Model classes_:", model.classes_)

    # Map label to string (adjust based on model.classes_)
    # Example: if classes_ = [0,1] and 1 means Fake, change accordingly.
    label = str(pred[0])
    # If your model uses 1 for Fake and 0 for Real, convert:
    try:
        if hasattr(model, "classes_"):
            # If classes_ are numeric, try mapping
            classes = list(model.classes_)
            print("classes list:", classes)
            # Heuristic: if classes are [0,1] assume 1->Fake
            if set(classes) == {0,1}:
                label = "Fake News" if pred[0] == 1 else "Real News"
    except Exception as e:
        print("Label mapping error:", e)

    return render_template('index.html', prediction=label)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)