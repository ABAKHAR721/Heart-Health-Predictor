from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import io
from PIL import Image
import base64
from models import User, engine, SessionLocal

app = Flask(__name__)

@app.route('/api/register', methods=['POST'])
def register():
    name = request.form['name']
    email = request.form['email']
    image = request.files['image']

    img = Image.open(io.BytesIO(image.read()))
    img = np.array(img)
    face_encodings = face_recognition.face_encodings(img)

    if face_encodings:
        face_embedding = face_encodings[0].tobytes()

        db = SessionLocal()
        user = User(name=name, email=email, face_embedding=face_embedding)
        db.add(user)
        db.commit()
        db.close()

        return jsonify({"status": "success", "message": "User registered successfully."})
    else:
        return jsonify({"status": "fail", "message": "No face detected in the image."}), 400

@app.route('/api/recognize', methods=['POST'])
def recognize():
    image_data = request.json['image']
    image_data = image_data.split(',')[1]  # Remove data URL prefix
    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
    img = np.array(img)

    face_encodings = face_recognition.face_encodings(img)

    if not face_encodings:
        return jsonify({"error": "No face detected"}), 400

    face_embedding = face_encodings[0].tobytes()

    db = SessionLocal()
    users = db.query(User).all()
    db.close()

    for user in users:
        known_embedding = np.frombuffer(user.face_embedding, dtype=np.float64)
        match = face_recognition.compare_faces([known_embedding], face_embedding)
        if match[0]:
            return jsonify({"name": user.name, "email": user.email}), 200

    return jsonify({"error": "No match found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
