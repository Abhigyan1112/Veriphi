from xmlrpc.client import Binary

from flask import Flask, url_for
import numpy as np
import io
from httpx._multipart import FileField
from mongoengine import connect, disconnect, Document, StringField, IntField, FileField
from flask import redirect, render_template, request, jsonify, get_flashed_messages
import cv2
import dlib
import qrcode
from pyzbar.pyzbar import decode

app=Flask(__name__)

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









# # set the Content-Type to multipart/form-data
# @app.route("/upload_images", methods=['POST'])
# def upload_images():
#     #Connecting to the MongoDB database 'attendees_images' with connection string
#     try:
#         connect(
#             db='attendees_images',
#             host='mongodb+srv://Abhigyan1112:veriphi123@verphi.mk8m4jf.mongodb.net/'
#         )
#     except Exception:
#         return jsonify({
#                 "status": "error",
#                 "message": "Cannot connect to the mongoDB server"
#         }), 400

#     try:
#         bookingID = request.form['bookingID']
#         names = request.form.getlist('names')
#         images = request.files.getlist('images')
#     except Exception:
#         return jsonify({
#             "status": "error",
#             "message": "Missing required fields: bookingID, names, or images"
#         }), 400












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
            host='mongodb+srv://Abhigyan1112:veriphi123@verphi.mk8m4jf.mongodb.net/'
        )

        attendees_for_bookingID = list(Entry.objects(bookingID=bookingID))
        disconnect(alias='default')

        # Connect to ticket IDs DB
        connect(
            db='ticket_IDs',
            host='mongodb+srv://Abhigyan1112:veriphi123@verphi.mk8m4jf.mongodb.net/',
        )

        # Pair each attendee with corresponding ticketID
        for attendee, ticketID in zip(attendees_for_bookingID, ticketIDs):

            img = qrcode.make(ticketID)
            img.save(f"{ticketID}-{bookingID}.png")
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




if __name__ == "__main__":
    app.run(debug=True,port=3000)