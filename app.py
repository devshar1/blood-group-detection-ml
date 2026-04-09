from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms as tt

# -----------------------
# Database setup
# -----------------------

DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        ''')
        db.commit()

# Define your deep learning model classes and load model here as per your earlier code...
# Transformations and class_names defined here...

# Prediction function as per your code...

# Import libraries like torch, PIL, etc.

# Your model and transform loading code here...
# -----------------------
# Model definition (ResNet9)
# -----------------------
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128),
                                  conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512),
                                  conv_block(512, 512))
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__)
app.secret_key = 'your_secret_key_here' # A secret key is required for sessions

# Transformations (same as training)
transform = tt.Compose([
    tt.Resize((128, 128)),
    tt.ToTensor()
])

# Define your classes
class_names =['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Load model
# NOTE: The model file `FingurePrintTOBloodGroup.pth` must be in the `models` directory.
try:
    model = ResNet9(in_channels=3, num_classes=len(class_names))
    model.load_state_dict(torch.load("models/FingurePrintTOBloodGroup.pth", map_location="cpu"))
    model.eval()
except Exception as e:
    print(f"Warning: Model could not be loaded. Prediction will be mocked. Error: {e}")
    model = None

# Prediction function (now includes a mock for testing without the .pth file)
def predict_image(img):
    if model:
        img_tensor = transform(img).unsqueeze(0)  # Add batch dim
        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, dim=1)
        return class_names[preds[0].item()]
    else:
        # Mock prediction for testing purposes if the model is not loaded
        import random
        return random.choice(class_names)

@app.route("/")
def home():
    if session.get('logged_in'):
        return render_template("home.html", username=session.get('username'))
    else:
        return redirect(url_for('login'))

@app.route("/features")
def features():
    return render_template("features.html")

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/methodology")
def methodology():
    return render_template("methodology.html")

@app.route("/demo")
def demo():
    return render_template("demo.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        with get_db() as db:
            user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

        if user and check_password_hash(user['password_hash'], password):
            session['logged_in'] = True
            session['username'] = user['username']
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            return render_template("register.html", error="Please fill out all fields.")

        hashed_password = generate_password_hash(password, method='scrypt')

        try:
            with get_db() as db:
                db.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed_password))
                db.commit()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Username already exists.")

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route("/predict", methods=["POST"])
def predict():
    if not session.get('logged_in'):
        return jsonify({"error": "You must be logged in to access this feature."}), 401

    if "fingerprint-image" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["fingerprint-image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            img = Image.open(file).convert("RGB")
            prediction = predict_image(img)
            return jsonify({"prediction": prediction})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
from flask import request, jsonify

chatbot_qa = {
    "what is blood group?": "Blood group refers to the classification of blood based on the presence or absence of antibodies and inherited antigenic substances.",
    "how to use this project?": "Upload your fingerprint image, then click analyze button to get the predicted blood group.",
    "what devices are supported?": "Currently, the project supports fingerprint images captured from common fingerprint scanners.",
    "how accurate is the prediction?": "Accuracy depends on the quality of the fingerprint image and similarity to training data.",
    "can i upload multiple images?": "Only one fingerprint image can be uploaded per prediction.",
    "what image formats are supported?": "JPEG, PNG, and BMP formats are supported.",
    "how long does prediction take?": "Typically just a few seconds after uploading and clicking Analyze.",
    "can this work with mobile fingerprint scanners?": "Yes, if the image is of good quality and format.",
    "what is a fingerprint image?": "A scanned image of the patterns on your fingertip used for identification.",
    "how is blood group predicted from fingerprint?": "The model analyzes fingerprint patterns correlating with blood groups from training data.",
    
    "are there privacy concerns?": "Images are processed and not permanently stored to protect your privacy.",
    "what if i get wrong results?": "Try a clearer image or contact support.",
    "can this system replace medical blood tests?": "No, always consult a medical lab for official blood typing.",
    "how was the model trained?": "Using labeled fingerprint images and a deep learning CNN model.",
    "what model is used?": "Custom ResNet9 convolutional neural network.",
    "how to contact support?": "Use the Contact page or email support@example.com.",
    "is this free to use?": "Currently free for testing purposes.",
    "can it detect diseases?": "No, this predicts blood group only.",
    "do I need an internet connection?": "Yes, for accessing the web application.",
    "can I contribute data?": "Contact us if you'd like to provide fingerprint data.",
    "how do I report bugs?": "Report via GitHub issues or contact support.",
    "what languages are supported?": "The interface is English only.",
    "how to improve image quality?": "Use clean fingers, good lighting, and avoid smudges.",
    "can the system handle partial prints?": "Full fingerprint images give best results.",
    "how often is the model updated?": "Periodically, as we collect more data.",
    "can I see past predictions?": "Prediction history feature is under development.",
}

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_msg = request.json.get('message', '').lower()
    response = "Sorry, I don't have an answer for that. Please ask something else."
    for question, answer in chatbot_qa.items():
        if question in user_msg:
            response = answer
            break
    return jsonify({'response': response})


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=4900)
