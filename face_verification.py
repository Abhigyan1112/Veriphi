import cv2
import face_recognition
import numpy as np
import time
import threading
from pyzbar.pyzbar import decode
from mongoengine import connect, Document, StringField, FileField
import gridfs
from pymongo import MongoClient

# --- Configuration and Database Setup ---
# NOTE: Replace with your actual MongoDB connection string
MONGO_URI = 'mongodb+srv://Abhigyan1112:veriphi123@verphi.mk8m4jf.mongodb.net/'

# Connection for MongoEngine (Primary DB access)
try:
    connect(db='ticket_IDs', alias='db_tickets', host=MONGO_URI)
    print("MongoDB (MongoEngine) connection successful.")
except Exception as e:
    print(f"Error connecting MongoEngine: {e}")
    exit()

# Connection for PyMongo/GridFS (File storage access)
try:
    client = MongoClient(MONGO_URI)
    db_tickets = client['ticket_IDs']
    fs = gridfs.GridFS(db_tickets)
    print("MongoDB (PyMongo/GridFS) connection successful.")
except Exception as e:
    print(f"Error connecting PyMongo: {e}")
    exit()

# --- Database Model (Must match your existing structure) ---
class ticketID_person(Document):
    ticketID = StringField(required=True, unique=True)
    bookingID = StringField(required=True)
    name = StringField(required=True)
    image = FileField(required=True)
    qr = FileField(required=True)
    meta = {'db_alias': 'db_tickets'}

# --- Global State Variables and Lock ---
# Lock to ensure safe access to shared state between the main loop and worker thread
state_lock = threading.Lock()

# Shared state variables
authorized_face_encoding = None
person_name = "Awaiting Scan"
scanned_ticket_id = None
worker_thread = None
is_processing = False
last_scan_time = 0
scan_cooldown = 3  # Seconds to wait before re-processing the same QR code

# --- Worker Function (Runs in a separate thread) ---
def db_worker(ticketID_to_process):
    """
    Handles the slow I/O and CPU-heavy tasks: DB lookup, GridFS fetch, and face encoding.
    This runs in a background thread and does not block the main video loop.
    """
    global authorized_face_encoding, person_name, is_processing

    print(f"[WORKER] Starting processing for TicketID: {ticketID_to_process}")
    
    # 1. DB Lookup
    entry = ticketID_person.objects(ticketID=ticketID_to_process).first()
    
    new_encoding = None
    new_name = None
    
    if entry:
        new_name = entry.name
        print(f"[WORKER] Found entry for {new_name}. Fetching image...")
        
        try:
            # 2. Image Fetch (GridFS)
            image_id = entry.image.grid_id
            image_file = fs.get(image_id)
            image_bytes = image_file.read()
            
            if image_bytes:
                # 3. Decode and Encode Database Image
                nparr = np.frombuffer(image_bytes, np.uint8)
                db_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if db_image is None:
                    raise Exception("cv2.imdecode failed to decode image")

                rgb_db_image = cv2.cvtColor(db_image, cv2.COLOR_BGR2RGB)
                
                # --- Optimization ---
                # Resize for faster encoding, but check image size first
                max_dim = max(rgb_db_image.shape[0], rgb_db_image.shape[1])
                scale = 600 / max_dim if max_dim > 600 else 1.0
                small_db_image = cv2.resize(rgb_db_image, (0, 0), fx=scale, fy=scale)
                
                print(f"[WORKER] DB Image resized to: {small_db_image.shape}")

                encodings = face_recognition.face_encodings(small_db_image)
                
                if encodings:
                    new_encoding = encodings[0]
                    print(f"[WORKER] Face encoding successful for {new_name}.")
                else:
                    print(f"[ERROR] No face found in database image for {new_name}")
                    new_name = f"{new_name} (NO FACE)"
            else:
                print(f"[ERROR] Image bytes empty for {new_name}")
                new_name = f"{new_name} (NO IMAGE)"

        except Exception as e:
            print(f"[ERROR] GridFS or Encoding failed: {e}")
            new_name = f"ERROR"
    else:
        print(f"[WORKER] TicketID {ticketID_to_process} is INVALID.")
        new_name = "INVALID TICKET"

    # 4. Update Shared State Safely
    with state_lock:
        authorized_face_encoding = new_encoding
        person_name = new_name
        is_processing = False
        
    print(f"[WORKER] Finished processing. Main loop updated.")


# --- Main Video Processing Loop ---
def run_verification_app():
    global scanned_ticket_id, worker_thread, is_processing, authorized_face_encoding, person_name, last_scan_time

    # Initialize video capture
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow("Real-Time Face Verification Scanner", cv2.WINDOW_AUTOSIZE)
    
    # Variables for resizing video feed for faster processing
    scale_factor = 0.5 

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Flip frame for a more natural mirror-like view
        frame = cv2.flip(frame, 1) 
        
        # Create a small frame for processing (QR and Face)
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # --- 1. QR Code Scanning ---
        # Only scan if we are not already processing a ticket
        if not is_processing:
            decoded_objects = decode(rgb_small_frame)
            for obj in decoded_objects:
                ticketID = obj.data.decode("utf-8")
                
                # Only process if it's a NEW ticket ID
                # or if the cooldown has passed for the SAME ticket ID
                current_time = time.time()
                if ticketID != scanned_ticket_id or (current_time - last_scan_time) > scan_cooldown:
                    scanned_ticket_id = ticketID
                    last_scan_time = current_time
                    is_processing = True # Set processing flag
                    
                    # Start the worker thread
                    worker_thread = threading.Thread(target=db_worker, args=(ticketID,))
                    worker_thread.start()
                    
                    # Immediately update state for user feedback
                    with state_lock:
                        person_name = "Processing..."
                        authorized_face_encoding = None
                    
                    break # Only process one QR code at a time
        
        # --- 2. Face Recognition ---
        # We hold local copies of the auth data so we don't need to hold the lock
        with state_lock:
            local_auth_encoding = authorized_face_encoding
            local_person_name = person_name
            local_is_processing = is_processing
            
        name_to_display = "Unknown"
        color = (0, 0, 255) # Red for Unknown

        if local_auth_encoding is not None:
            # Find faces in the small, processed frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            current_face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            STRICT_TOLERANCE = 0.5

            for (top, right, bottom, left), face_encoding in zip(face_locations, current_face_encodings):
                matches = face_recognition.compare_faces([local_auth_encoding], face_encoding, tolerance = STRICT_TOLERANCE)
                
                if True in matches:
                    name_to_display = f"{local_person_name} (MATCH)"
                    color = (0, 255, 0) # Green for Match
                
                # --- Draw on the ORIGINAL, full-sized frame ---
                # Scale coordinates back up
                top = int(top / scale_factor)
                right = int(right / scale_factor)
                bottom = int(bottom / scale_factor)
                left = int(left / scale_factor)

                # Draw the box
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name_to_display, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # --- 3. Display Status ---
        status_text = f"Scanned: {local_person_name}"
        
        if local_is_processing:
            status_color = (0, 165, 255) # Orange for "Processing"
        elif "INVALID" in local_person_name or "ERROR" in local_person_name or "NO FACE" in local_person_name:
            status_color = (0, 0, 255) # Red for Invalid
        elif local_auth_encoding is not None:
            status_color = (0, 255, 0) # Green for Valid/Loaded
        else:
            status_color = (255, 255, 255) # White for "Awaiting Scan"

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        # --- 4. Show Frame ---
        cv2.imshow("Real-Time Face Verification Scanner", frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    camera.release()
    cv2.destroyAllWindows()
    print("Application stopped.")
    # Ensure worker thread can exit if app closes
    if worker_thread and worker_thread.is_alive():
        print("Waiting for worker thread to finish...")
        is_processing = False # Signal thread (though it should finish on its own)
        worker_thread.join()
    print("Cleanup complete.")

if __name__ == '__main__':
    run_verification_app()
