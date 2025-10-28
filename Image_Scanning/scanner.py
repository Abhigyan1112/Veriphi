from xmlrpc.client import Binary

from flask import Flask, url_for
import numpy as np
import io
from httpx._multipart import FileField
from mongoengine import connect, disconnect, Document, StringField, IntField, FileField
from flask import redirect, render_template, request, jsonify, get_flashed_messages
import cv2
import mediapipe as mp
import qrcode
from pyzbar.pyzbar import decode
import os 
from flask_cors import CORS
import math


print("--- SERVER RESTARTED WITH CORS ENABLED ---")

app=Flask(__name__)
CORS(app)
mongodb_uri = os.environ.get('MONGODB_URI')

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)


class Entry(Document):
    bookingID=StringField(required=True)
    imageID=IntField(required=True)
    name=StringField(required=True)
    image=FileField(required=True)

class ticketID_person(Document):
    ticketID = StringField(required=True, unique=True)   # use ticketID instead of bookingID
    name = StringField(required=True)
    image = FileField(required=True)
    qr = FileField(required=True)


@app.route("/")
def main():
    return render_template("index.html")




@app.route("/health")
def health_check():
    return jsonify({"status": "ok"}), 200






@app.route("/frontal_check", methods=['POST'])
def image_processing():
    """
    This endpoint checks if an uploaded image contains a single, front-facing human face.
    """
    global face_mesh # <-- ADD THIS LINE
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400

    image_file = request.files['image']
    image_data = image_file.read()

    # Decode image from buffer
    img_np = np.frombuffer(image_data, np.uint8)
    img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img_cv2 is None:
        return jsonify({"status": "error", "message": "Failed to load image."}), 400

    try:
        # Get image dimensions
        image_height, image_width, _ = img_cv2.shape

        # Convert the BGR image to RGB (Corrected constant: COLOR_BGR2RGB)
        rgb_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        
        # Process the image with FaceMesh
        results = face_mesh.process(rgb_image)

        # Check if any faces were detected
        if not results.multi_face_landmarks:
            return jsonify({"status": "error", "message": "No faces detected."}), 400
        
        # --- Orientation Analysis ---
        # We already set max_num_faces=1, so we can safely take the first result.
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Key landmarks for orientation calculation (using the official MediaPipe landmark map)
        left_eye = face_landmarks[33]      # Left eye outer corner
        right_eye = face_landmarks[263]    # Right eye outer corner
        nose_tip = face_landmarks[1]       # Nose tip
        nose_bridge = face_landmarks[6]    # Top of nose bridge, for pitch
        chin = face_landmarks[152]         # Bottom of the chin, for pitch

        # Convert normalized coordinates to pixel coordinates
        le_x, le_y = int(left_eye.x * image_width), int(left_eye.y * image_height)
        re_x, re_y = int(right_eye.x * image_width), int(right_eye.y * image_height)
        nose_x = int(nose_tip.x * image_width)
        nose_bridge_y = int(nose_bridge.y * image_height)
        nose_tip_y = int(nose_tip.y * image_height)
        chin_y = int(chin.y * image_height)

        # 1. Calculate ROLL (head tilt)
        # We check the vertical difference between the eyes. A large difference means the head is tilted.
        delta_y = re_y - le_y
        delta_x = re_x - le_x
        roll_angle = math.degrees(math.atan2(delta_y, delta_x))

        # 2. Calculate YAW (side-to-side turn)
        # We check if the nose is centered between the eyes.
        eye_center_x = (le_x + re_x) // 2
        # Use abs() to ensure distance is positive, preventing sqrt domain errors
        interocular_distance = math.sqrt(abs(delta_x**2 + delta_y**2))
        
        # Ratio of nose offset from center to the distance between eyes
        # A small ratio means the nose is well-centered.
        nose_offset_ratio = abs(nose_x - eye_center_x) / interocular_distance if interocular_distance else 0

        # 3. Calculate PITCH (looking up/down)
        # We compare the vertical distance from nose bridge to tip with the distance from nose tip to chin.
        # A large difference indicates the head is tilted up or down.
        upper_face_height = nose_tip_y - nose_bridge_y
        lower_face_height = chin_y - nose_tip_y
        pitch_ratio = upper_face_height / lower_face_height if lower_face_height != 0 else 1.0
        
        # --- Stricter Decision Thresholds ---
        # You can adjust these values for stricter or more lenient checks.
        ROLL_THRESHOLD = 10.0   # Stricter: Max 10 degrees of tilt (was 15.0)
        YAW_THRESHOLD = 0.12    # Very Strict: Nose must be almost perfectly centered (was 0.20)
        PITCH_RATIO_RANGE = (0.35, 0.75) # Stricter: Head must be more level (was 0.8, 1.2)

        if abs(roll_angle) > ROLL_THRESHOLD:
            return jsonify({
                "status": "error",
                "message": f"Face is tilted too much. (Roll: {roll_angle:.2f}°)"
            }), 400

        if nose_offset_ratio > YAW_THRESHOLD:
            return jsonify({
                "status": "error",
                "message": f"Face is not looking forward. (Yaw Ratio: {nose_offset_ratio:.2f})"
            }), 400

        if not (PITCH_RATIO_RANGE[0] <= pitch_ratio <= PITCH_RATIO_RANGE[1]):
            return jsonify({
                "status": "error",
                "message": f"Face is looking up or down. (Pitch Ratio: {pitch_ratio:.2f})"
            }), 400

        # If all checks pass, the face is considered frontal
        return jsonify({
            "status": "success",
            "message": "Valid frontal face detected"
        }), 200
            
    except Exception as e:
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500










# set the Content-Type to multipart/form-data
@app.route("/upload_images", methods=['POST'])
def upload_images():
    #Connecting to the MongoDB database 'attendees_images' with connection string
    try:
        connect(
            db='attendees_images',
            host=mongodb_uri
        )
    except Exception:
        return jsonify({
                "status": "error",
                "message": "Cannot connect to the mongoDB server"
        }), 400

    try:
        bookingID = request.form['bookingID']
        names = request.form.getlist('names')
        images = request.files.getlist('images')
    except Exception:
        return jsonify({
            "status": "error",
            "message": "Missing required fields: bookingID, names, or images"
        }), 400

    if len(names) != len(images):
        return jsonify({
            "status": "error",
            "message": f"Mismatch: Received {len(names)} names and {len(images)} images. They must be equal."
        }), 400
        
    if len(names) == 0:
        return jsonify({"status": "error", "message": "No names or images received"}), 400

    saved_entries = []
    try:
        for i, (name, image_file) in enumerate(zip(names, images)):
            if image_file.filename == '':
                continue 

            entry = Entry(
                bookingID=bookingID,
                imageID=i + 1,
                name=name
            )
            
            entry.image.put(image_file.stream, content_type=image_file.content_type)
            entry.save()
            saved_entries.append(name)

        return jsonify({
            "status": "success",
            "message": f"Successfully uploaded images for {len(saved_entries)} people under bookingID {bookingID}",
            "bookingID": bookingID,
            "names_processed": saved_entries
        }), 201

    except Exception as e:
        print(f"An error occurred during upload: {e}") 
        return jsonify({"status": "error", "message": str(e)}), 500














@app.route("/qr_generation", methods=['POST'])
def QR_generation():
    try:
        data = request.get_json()
        bookingID = data.get("bookingID")
        ticketIDs = data.get("ticketIDs")
        
        if not bookingID or not ticketIDs:
            return jsonify({
                "status": "error",
                "message": "bookingID and ticketIDs are required"
            }), 400

        connect(
            db='attendees_images',
            host=mongodb_uri
        )

        attendees_for_bookingID = list(Entry.objects(bookingID=bookingID))
        disconnect(alias='default')

        # Connect to ticket IDs DB
        connect(
            db='ticket_IDs',
            host=mongodb_uri
        )

        # Pair each attendee with corresponding ticketID
        for attendee, ticketID in zip(attendees_for_bookingID, ticketIDs):

            img = qrcode.make(ticketID)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

            ticketID_entry = ticketID_person(
                ticketID=ticketID,
                name=attendee.name,
                image=attendee.image,
                qr = buffer.getvalue()
            )
            ticketID_entry.save()

        disconnect(alias='default')

        return jsonify({
            "status": "success",
            "message": f"QR generated for bookingID {bookingID}"
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)