from xmlrpc.client import Binary

from flask import Flask, url_for
import numpy as np
from httpx._multipart import FileField
from mongoengine import connect, Document, StringField, IntField, FileField
from flask import redirect, render_template, request, jsonify, flash, get_flashed_messages
from flask_sqlalchemy import SQLAlchemy
import cv2
import dlib

app=Flask(__name__)
app.secret_key = 'aditi18'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bookings.db'
db=SQLAlchemy(app)


class Image(db.Model):
    booking_id=db.Column(db.String(200),nullable=False)
    image_id=db.Column(db.Integer,primary_key=True,autoincrement=True)
    name=db.Column(db.String(200),nullable=False)
    image=db.Column(db.LargeBinary,nullable=False)

    def __repr__(self) -> str:
        return f'{self.booking_id} - {self.image_id} - {self.name}'

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
    try:
        all_bookings = db.session.query(Image.booking_id).distinct().all()
    except Exception as e:
        db.session.rollback()
        flash(str(e), 'error')
    booking_id=request.args.get('booking_id')
    messages=get_flashed_messages(with_categories=True)
    return render_template("index.html",all_bookings=all_bookings,dropdown_placeholder=booking_id or "Drop Down", messages=messages)

@app.route("/image-processing",methods=['POST'])
def image_processing():
    try:
        booking_id=request.form['bookingID']
        name=request.form['name']
    except Exception:
        flash("bookingID or Name not provided","error")
        return redirect(url_for("main"))

    # Corrected check: Verify the file exists in the request first.
    if 'image' not in request.files or request.files['image'].filename == '':
        flash("No file selected","error")
        return redirect(url_for("main"))

    # Read the file data *after* verifying it exists
    image_file = request.files['image']
    image_data = image_file.read()

    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("static/shape_predictor_68_face_landmarks.dat")

        img_np = np.frombuffer(image_data, np.uint8)
        img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img_cv2 is None:
            return jsonify({'error': 'Failed to load image with OpenCV. Check format.'}), 400

        rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        faces = detector(rgb)

        if len(faces) == 0:
            return jsonify({
                'status': 'Cannot detect any faces in the image or the face is not frontal'
            }), 400
        if len(faces) > 1:
            return jsonify({
                'status': 'Multiple faces in the image'
            }), 400
        
        # All processing logic remains the same...
        landmarks = predictor(rgb, faces[0])
        points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

        for (x, y) in points:
            cv2.circle(img_cv2, (x, y), 2, (255, 0, 0), -1)

        nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_cheek = (landmarks.part(1).x, landmarks.part(1).y)
        right_cheek = (landmarks.part(15).x, landmarks.part(15).y)

        left_dist = abs(nose[0] - left_cheek[0])
        right_dist = abs(right_cheek[0] - nose[0])
        
        # Avoid division by zero
        if right_dist == 0:
             return jsonify({
                'status': 'Invalid face landmarks for symmetry check.'
            }), 400

        symmetry_ratio = left_dist / right_dist

        if symmetry_ratio > 1.5 or symmetry_ratio < (1/1.5): # Check for both left and right turns
            return jsonify({
                'status': 'The image is not frontal',
                'symmetry_ratio': symmetry_ratio
            }), 400

        # Save the original image data, not the processed one with circles
        img = Image(booking_id=booking_id, name=name, image=image_data)
        db.session.add(img)
        db.session.commit()
        
        flash("Image uploaded and processed successfully!", "success")
        return redirect(url_for('main'))

    except Exception as e:
        return jsonify({'error Exception': str(e)}), 500

@app.route("/dropdown/<string:booking_id>",methods=['GET'])
def dropdown(booking_id):
    return redirect(url_for("main",booking_id=booking_id))

@app.route("/entry/<string:booking_id>",methods=['GET'])
def entry(booking_id):
    if booking_id == "Drop Down":
        flash("select a booking id","error")
        return redirect(url_for("main"))
    images=Image.query.filter(Image.booking_id==booking_id).all()
    if not images:
        flash("No images found for the selected bookingID","error")
        return redirect(url_for("main"))
    for img in images:
        mongo_image=Entry(bookingID=booking_id,name=img.name,image=img.image,imageID=img.image_id)
        try:
            mongo_image.save()
            db.session.delete(img)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            flash(str(e), 'error')
    return redirect(url_for('main'))

#   GET request to return a form

@app.route("/show",methods=['GET'])
def show():
    users = Entry.objects().to_json()
    return users, 200

if __name__ == "__main__":
    app.run(debug=True,port=3000)