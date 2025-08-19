from xmlrpc.client import Binary

from flask import Flask, url_for
import numpy as np
from httpx._multipart import FileField
from mongoengine import connect, Document, StringField, IntField, FileField
from flask import redirect, render_template, request, jsonify, get_flashed_messages
import cv2
import dlib

app=Flask(__name__)

#Connecting to the MongoDB database 'trial_database' with connection string
connect(
    db='attendees_images',
    host='mongodb+srv://Abhigyan1112:veriphi123@verphi.mk8m4jf.mongodb.net/'
)

class Entry(Document):
     bookingID=StringField(required=True)
     imageID=IntField(required=True)
     name=StringField(required=True)
     image=FileField(required=True)

@app.route("/")
def main():
    booking_id=request.args.get('booking_id')
    messages=get_flashed_messages(with_categories=True)
    return render_template("index.html",dropdown_placeholder=booking_id or "Drop Down", messages=messages)

@app.route("/image-processing", methods=['POST'])
def image_processing():
    # Ensure file exists in request
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400

    image_file = request.files['image']
    image_data = image_file.read()

    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("static/shape_predictor_68_face_landmarks.dat")

        img_np = np.frombuffer(image_data, np.uint8)
        img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img_cv2 is None:
            return jsonify({
                "status": "error",
                "message": "Failed to load image with OpenCV. Check format."
            }), 400

        rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        faces = detector(rgb)

        if len(faces) == 0:
            return jsonify({
                "status": "error",
                "message": "No faces detected or face not frontal"
            }), 400
        if len(faces) > 1:
            return jsonify({
                "status": "error",
                "message": "Multiple faces detected"
            }), 400

        # Extract landmarks
        landmarks = predictor(rgb, faces[0])
        points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

        nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_cheek = (landmarks.part(1).x, landmarks.part(1).y)
        right_cheek = (landmarks.part(15).x, landmarks.part(15).y)

        left_dist = abs(nose[0] - left_cheek[0])
        right_dist = abs(right_cheek[0] - nose[0])

        if right_dist == 0:
            return jsonify({
                "status": "error",
                "message": "Invalid face landmarks for symmetry check."
            }), 400

        symmetry_ratio = left_dist / right_dist

        if symmetry_ratio > 1.5 or symmetry_ratio < (1/1.5):
            return jsonify({
                "status": "error",
                "message": "Face is not frontal",
                "symmetry_ratio": symmetry_ratio
            }), 400

        # ✅ Success response
        return jsonify({
            "status": "success",
            "message": "Valid frontal face detected",
            "symmetry_ratio": symmetry_ratio
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True,port=3000)