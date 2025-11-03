from xmlrpc.client import Binary

from flask import Flask, url_for
import numpy as np
import io
import json
from httpx._multipart import FileField
from mongoengine import connect, disconnect, Document, StringField, IntField, FileField
from flask import redirect, render_template, request, jsonify, get_flashed_messages
import base64
import cv2
import mediapipe as mp
import qrcode
from pyzbar.pyzbar import decode
import os 
from flask_cors import CORS
import math
import urllib



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
    bookingID = StringField(required=True)
    name = StringField(required=True)
    image = FileField(required=True)
    qr = FileField(required=True)

    meta = {'collection': 'ticket_i_d_person'}

    def to_dict_with_files(self):
        """
        Custom method to serialize document and encode files to Base64.
        """
        image_data = None
        qr_data = None
        
        # Read the image file from GridFS and encode it
        if self.image:
            try:
                # .read() gets the binary data
                image_binary = self.image.read()
                # .b64encode() returns bytes, so we .decode('utf-8') to get a string
                image_data = base64.b64encode(image_binary).decode('utf-8')
            except Exception as e:
                print(f"Error reading image for {self.ticketID}: {e}")
                image_data = None

        # Read the QR file from GridFS and encode it
        if self.qr:
            try:
                qr_binary = self.qr.read()
                qr_data = base64.b64encode(qr_binary).decode('utf-8')
            except Exception as e:
                print(f"Error reading QR for {self.ticketID}: {e}")
                qr_data = None

        # Return a JSON-friendly dictionary
        return {
            "_id": str(self.id),
            "ticketID": self.ticketID,
            "bookingID": self.bookingID,
            "name": self.name,
            # We add a data URI prefix so it can be used directly in an <img src>
            "image_base64": f"data:image/png;base64,{image_data}" if image_data else None,
            "qr_base64": f"data:image/png;base64,{qr_data}" if qr_data else None,
        }


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
    try:
        connect(db='attendees_images', host=mongodb_uri)
    except Exception:
        return jsonify({
            "status": "error",
            "message": "Cannot connect to the MongoDB server"
        }), 400

    try:
        # ✅ Debug: check what we actually received
        print("🔍 Received form fields:", list(request.form.keys()))
        print("🔍 Received files:", list(request.files.keys()))

        # ✅ Try to read metadata as form field first
        metadata_str = request.form.get('metadata')

        # Fallback: sometimes metadata arrives as a file
        if not metadata_str and 'metadata' in request.files:
            metadata_str = request.files['metadata'].read().decode('utf-8')

        # Still missing metadata? Return error
        if not metadata_str:
            return jsonify({
                "status": "error",
                "message": "Missing metadata field"
            }), 400

        # ✅ Decode URI-encoded JSON (Flutter encodes it)
        metadata_str = urllib.parse.unquote(metadata_str)

        # ✅ Parse JSON safely
        metadata = json.loads(metadata_str)
        bookingID = metadata.get('bookingID')
        names = metadata.get('attendees', [])

        if not bookingID or not names:
            return jsonify({
                "status": "error",
                "message": "Missing bookingID or attendees list"
            }), 400

        # ✅ Get uploaded images
        images = request.files.getlist('images')

        if len(names) != len(images):
            return jsonify({
                "status": "error",
                "message": f"Mismatch: received {len(names)} names but {len(images)} images"
            }), 400

        saved_entries = []
        for i, (name, image_file) in enumerate(zip(names, images)):
            if not image_file or image_file.filename == '':
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
            "message": f"Successfully uploaded {len(saved_entries)} images for bookingID {bookingID}",
            "bookingID": bookingID,
            "names_processed": saved_entries
        }), 201

    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500














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

        # --- Step 1: Connect to source DB and READ file data into memory ---
        connect(
            db='attendees_images',
            host=mongodb_uri,
            alias='source_db'
        )
        attendees_from_source = list(Entry.objects(bookingID=bookingID))

        # We MUST read the image data into memory before disconnecting
        attendee_data_list = []
        for att in attendees_from_source:
            image_binary = None
            try:
                if att.image:
                    image_binary = att.image.read() # Read data while connected
            except Exception as e:
                print(f"Error reading source image for {att.name}: {e}")
            
            attendee_data_list.append({
                "name": att.name,
                "image_binary": image_binary # Store the raw bytes
            })

        disconnect(alias='source_db') # Disconnect from 'attendees_images'

        # Check for count mismatch before proceeding
        if len(attendee_data_list) != len(ticketIDs):
             return jsonify({
                "status": "error",
                "message": f"Data mismatch: Found {len(attendee_data_list)} attendees but received {len(ticketIDs)} ticketIDs."
            }), 400

        # --- Step 2: Connect to destination DB and WRITE new docs with raw data ---
        connect(
            db='ticket_IDs',
            host=mongodb_uri,
            alias='destination_db'
        )

        # Pair each attendee's data with the corresponding ticketID
        for attendee_data, ticketID in zip(attendee_data_list, ticketIDs):

            img = qrcode.make(ticketID)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0) # Go to the start of the buffer

            ticketID_entry = ticketID_person(
                ticketID=ticketID,
                bookingID=bookingID,
                name=attendee_data["name"],
                image=attendee_data["image_binary"], # <-- Use the raw bytes
                qr = buffer.getvalue() # <-- Use the raw bytes
            )
            ticketID_entry.save()

        disconnect(alias='destination_db') # Disconnect from 'ticket_IDs'

        return jsonify({
            "status": "success",
            "message": f"QR generated for bookingID {bookingID}"
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


    

@app.route("/get_tickets", methods=['POST'])
def get_tickets():
    try:

        if "bookingID" not in request.form:
            return jsonify({
                "status": "error",
                "message": "Missing 'bookingID' in form data."
            }), 400
        
        connect(
            db='ticket_IDs',
            host=mongodb_uri
        )
        
        bookingID = request.form["bookingID"]

        query_filter = {"bookingID" : bookingID}
        found_tickets = ticketID_person.objects(bookingID=bookingID)
        tickets_list = [ticket.to_dict_with_files() for ticket in found_tickets]

        return jsonify({
            "status": "success",
            "count": len(tickets_list),
            "tickets": tickets_list
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500






@app.route("/resale_tickets", method=['POST'])
def resale_tickets():
    try:


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)