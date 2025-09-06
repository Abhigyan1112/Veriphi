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


app=Flask(__name__)
mongodb_uri = os.environ.get('MONGODB_URI')

face_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
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










@app.route("/frontal_check", methods=['POST'])
def image_processing():
    # Ensure file exists in request
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400

    image_file = request.files['image']
    image_data = image_file.read()
    
    img_np = np.frombuffer(image_data, np.uint8)
    img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img_cv2 is None:
        return jsonify({"status": "error", "message": "Failed to load image."}), 400
    
    try:
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR_RGB)
        
        # Use the globally initialized face_detector
        results = face_detector.process(rgb_image)

        if not results.detections:
            return jsonify({"status": "error", "message": "No faces detected."}), 400
        
        if len(results.detections) > 1:
            return jsonify({"status": "error", "message": "Multiple faces detected."}), 400
        
        return jsonify({
            "status": "success",
            "message": "Valid face detected"
        }), 200
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500










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

