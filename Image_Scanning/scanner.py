from flask import Flask
from mongoengine import connect, Document, StringField, IntField
from flask import redirect, render_template, request, jsonify
import cv2
import os
import dlib

app=Flask(__name__)

# #Connecting to the MongoDB database 'trial_database' with connection string
# connect(
#     db='trial_database',
#     host='mongodb+srv://Abhigyan:%40Saransh16@user-images.48yzbzo.mongodb.net/'
# )

# class Entry(Document):
#     name=StringField(required=True)
#     age= IntField(required=True)
#     email=StringField(required=True)

@app.route("/")
def main():
    return "Welcome to the home page"

@app.route("/image-processing",methods=['GET'])
def image_processing():
    path='static/Images'
    opath='static/marked_Images'
    result=[]

    for image in os.listdir(path):
        image_path=os.path.join(path,image)
        try:
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("static/shape_predictor_68_face_landmarks.dat")

            img=cv2.imread(image_path)

            if img is None:
                return jsonify({'error': 'Failed to load image with OpenCV. Check format or path.'}), 400

            
            rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            faces=detector(rgb)

            if len(faces) == 0:
                result.append({
                    'file_name': image,
                    'status': 'Cannot detect any faces in the image'
                })
                continue

            for face in faces:
                landmarks = predictor(rgb, face)
                points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

                for (x,y) in points:
                    cv2.circle(img, (x,y), 2, (255,0,0), -1)

                nose = (landmarks.part(30).x, landmarks.part(30).y)
                left_cheek = (landmarks.part(1).x, landmarks.part(1).y)
                right_cheek = (landmarks.part(15).x, landmarks.part(15).y)

                left_dist = abs(nose[0] - left_cheek[0])
                right_dist = abs(right_cheek[0] - nose[0])
                symmetry_ratio = left_dist / right_dist


            output_path = os.path.join(opath,('marked_'+image))
            cv2.imwrite(output_path, img)

            if symmetry_ratio > 2:
                result.append({
                    'file_name': image,
                    'status': 'The image is not frontal',
                    'smmetry_ratio': symmetry_ratio
                })
                continue

            result.append({
                'file_name': image,
                'output_image': output_path,
                'Symmetry_ratio' : symmetry_ratio
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'results': result}), 200

# @app.route("/entry",methods=['GET','POST'])
# def entry():
#     if request.method=='POST':
#         name=request.form['name']
#         age=request.form['age']
#         email=request.form['email']
#         entry=Entry(name=name,age=age,email=email)
#         entry.save()
#         return f"<h3>Entry Added! ID:{entry.id}</h3> <a href='/entry'>Add Another?</a>"
    
# #   GET request to return a form
#     return render_template('index.html')

# @app.route("/show",methods=['GET'])
# def show():
#     users = Entry.objects().to_json()
#     return users, 200

if __name__ == "__main__":
    app.run(debug=True,port=3000)