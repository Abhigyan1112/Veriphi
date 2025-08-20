from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
from pyzbar.pyzbar import decode
from mongoengine import connect, Document, StringField, FileField
import gridfs # <-- IMPORT THIS
from pymongo import MongoClient # <-- AND THIS

app = Flask(__name__)

# --- Database Connection and Model ---
class ticketID_person(Document):
    ticketID = StringField(required=True, unique=True)
    name = StringField(required=True)
    image = FileField(required=True)
    qr = FileField(required=True)
    meta = {'db_alias': 'db_tickets'} # This model uses the ticket_IDs DB

# --- MODIFICATION START ---
# Connect to BOTH databases
MONGO_URI = 'mongodb+srv://Abhigyan1112:veriphi123@verphi.mk8m4jf.mongodb.net/'

# Connection for MongoEngine (our primary DB)
connect(db='ticket_IDs', alias='db_tickets', host=MONGO_URI)

# Connection for PyMongo/GridFS (to fetch images from the other DB)
client = MongoClient(MONGO_URI)
db_attendees = client['attendees_images'] # The name of your other database
fs = gridfs.GridFS(db_attendees)
# --- MODIFICATION END ---


# --- Global variables ---
camera = cv2.VideoCapture(0)
authorized_face_encoding = None
person_name = None
scanned_ticket_id = None

def generate_video_stream():
    global authorized_face_encoding, person_name, scanned_ticket_id

    while True:
        success, frame = camera.read()
        if not success:
            break

        # --- 1. QR Code Scanning ---
        decoded_objects = decode(frame)
        for obj in decoded_objects:
            ticketID = obj.data.decode("utf-8")

            if ticketID != scanned_ticket_id:
                scanned_ticket_id = ticketID
                entry = ticketID_person.objects(ticketID=ticketID).first()

                if entry:
                    person_name = entry.name
                    print(f"[VALID] TicketID: {ticketID}, Name: {person_name}")

                    # --- MODIFICATION START ---
                    # The original way of reading the image is broken.
                    # image_bytes = entry.image.read() # <-- THIS IS THE BROKEN LINE

                    # NEW WAY: Manually fetch the image from the OTHER database using GridFS.
                    image_id = entry.image.grid_id # Get the ID of the image file
                    image_file = fs.get(image_id)   # Find that ID in the attendees_images DB
                    image_bytes = image_file.read() # Read the binary data
                    # --- MODIFICATION END ---

                    if image_bytes:
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        db_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        rgb_db_image = cv2.cvtColor(db_image, cv2.COLOR_BGR2RGB)
                        encodings = face_recognition.face_encodings(rgb_db_image)
                        
                        if encodings:
                            authorized_face_encoding = encodings[0]
                        else:
                            authorized_face_encoding = None
                            person_name = None
                            print(f"[ERROR] No face found in database image for {entry.name}")
                    else:
                         authorized_face_encoding = None

                else:
                    print(f"[INVALID] TicketID: {ticketID}")
                    authorized_face_encoding = None
                    person_name = None
                    scanned_ticket_id = "INVALID"
        
        # --- (The rest of the face verification code is unchanged) ---
        if authorized_face_encoding is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            current_face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, current_face_encodings):
                matches = face_recognition.compare_faces([authorized_face_encoding], face_encoding)
                name_to_display = "Unknown"
                color = (0, 0, 255)
                if True in matches:
                    name_to_display = f"{person_name} (MATCH)"
                    color = (0, 255, 0)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name_to_display, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        if scanned_ticket_id:
            status_text = f"Scanned: {person_name if person_name else scanned_ticket_id}"
            status_color = (0, 255, 0) if person_name else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_stream')
def video_stream():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5001)