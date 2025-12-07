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
    # form मधल्या "news" field मधून text घे
    text = request.form['news']
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

    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vect)
    except Exception as e:
        print("predict_proba error:", e)

    print("=== DEBUG PREDICTION ===")
    print("Input text:", text_proc)
    print("Vector shape:", getattr(vect, "shape", None))
    print("Model type:", type(model))
    print("Prediction raw:", pred)
    if proba is not None:
        print("Predict_proba:", proba)
    if hasattr(model, "classes_"):
        print("Model classes_:", model.classes_)

    label = str(pred[0])

    try:
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            print("classes list:", classes)
            if set(classes) == {0, 1}:
                label = "Fake News" if pred[0] == 1 else "Real News"
    except Exception as e:
        print("Label mapping error:", e)

    # इथे आपण prediction नावाने template ला पाठवतो
    return render_template('index.html', prediction=label)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
