from flask import Flask, request, render_template, redirect, send_from_directory, url_for, jsonify, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import base64
import os
import io
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
from datetime import datetime
import uuid
warnings.filterwarnings("ignore")


import argparse
from os.path import join
import dlib
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.secret_key = 'xyzsdfg'

mysql = MySQL(app)
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'deeptech'

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint_path = os.path.join(script_dir, 'resnetinceptionv1_epoch_32.pth')

# Load the checkpoint file
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

@app.route('/index')
def index():
    if 'loggedin' in session and session['loggedin']:
        user_name = session.get('name', '')
        user_logged_in = session.get('loggedin', False)
        return render_template('index.html', user_name=user_name, user_logged_in=user_logged_in)
    else:
        return render_template('index.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/image')
def image():
    if 'loggedin' in session and session['loggedin']:
        user_name = session.get('name', '')
        user_logged_in = session.get('loggedin', False)
        return render_template('image.html', user_name=user_name, user_logged_in=user_logged_in)
    else:
        return render_template('image.html')

@app.route('/video')
def video():
    if 'loggedin' in session and session['loggedin']:
        user_name = session.get('name', '')
        user_logged_in = session.get('loggedin', False)
        return render_template('video.html', user_name=user_name, user_logged_in=user_logged_in)
    else:
        return render_template('video.html')

@app.route('/detector')
def detector():
    if 'loggedin' in session and session['loggedin']:
        user_name = session.get('name', '')
        user_logged_in = session.get('loggedin', False)
        return render_template('detector.html', user_name=user_name, user_logged_in=user_logged_in)
    else:
        return render_template('detector.html')

@app.route('/about')
def about():
    if 'loggedin' in session and session['loggedin']:
        user_name = session.get('name', '')
        user_logged_in = session.get('loggedin', False)
        return render_template('about.html', user_name=user_name, user_logged_in=user_logged_in)
    else:
        return render_template('about.html')

@app.route('/contact')
def contact():
    if 'loggedin' in session and session['loggedin']:
        user_name = session.get('name', '')
        user_logged_in = session.get('loggedin', False)
        return render_template('contact.html', user_name=user_name, user_logged_in=user_logged_in)
    else:
        return render_template('contact.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('userid', None)
    session.pop('email', None)
    return render_template('login.html')
  
@app.route('/signup', methods =['GET', 'POST'])
def register():
    message = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form:
        userName = request.form['name']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        account = cursor.fetchone()
        if account:
            message = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            message = 'Invalid email address!'
        elif not userName or not password or not email:
            message = 'Please fill out the form!'
        else:
            cursor.execute('INSERT INTO users VALUES (NULL, %s, %s, %s)', (userName, email, password,))
            mysql.connection.commit()
            message = 'You have successfully registered!'
            return render_template('login.html', message=message)  # Redirect to login with message
    elif request.method == 'POST':
        message = 'Please fill out the form!'
    return render_template('signup.html', message=message)


@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = % s AND password = % s', (email, password, ))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['id'] = user['id']
            session['name'] = user['name']
            session['email'] = user['email']
            message = 'Logged in successfully !'
            return redirect(url_for('homeLogin', message=message))
        else:
            message = 'Please enter correct email / password !'
    return render_template('login.html', message=message)

@app.route('/loginHome')
def homeLogin():
    if 'loggedin' in session and session['loggedin']:
        user_name = session.get('name', '')
        user_logged_in = session.get('loggedin', False)
        message = f'Welcome, {session["name"]}!'
        return render_template('index.html', message=message, user_name=user_name, user_logged_in=user_logged_in)
    else:
        return render_template('index.html')


@app.route('/faq')
def faq():
    if 'loggedin' in session and session['loggedin']:
        user_name = session.get('name', '')
        user_logged_in = session.get('loggedin', False)
        return render_template('faq.html', user_name=user_name, user_logged_in=user_logged_in)
    else:
        return render_template('faq.html')

@app.route('/resultImage/<int:detection_id>')
def resultImage(detection_id):
    if 'loggedin' in session and session['loggedin']:
        user_name = session.get('name', '')
        user_logged_in = session.get('loggedin', False)
        
        # Fetch the detection data from the database based on the detection_id
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM detection WHERE id = %s", (detection_id,))
        detection_data = cursor.fetchone()
        
        # Extract necessary data from the detection_data
        real_confidence = detection_data[1]
        fake_confidence = detection_data[2]
        prediction_class = detection_data[3]
        image_path = detection_data[4]

        # Pass the data to the template
        return render_template('resultImage.html', user_name=user_name, user_logged_in=user_logged_in,
                               real_confidence=real_confidence, fake_confidence=fake_confidence,
                               prediction_class=prediction_class, image_path=image_path)
    else:
        return render_template('resultImage.html')

@app.route('/resultVideo/<int:detection_id>')
def resultVideo(detection_id):
    if 'loggedin' in session and session['loggedin']:
        user_name = session.get('name', '')
        user_logged_in = session.get('loggedin', False)

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM detectionvideo WHERE id = %s", (detection_id,))
        detection_data = cursor.fetchone()

        real_confidence = detection_data[1]
        fake_confidence = detection_data[2]
        overall_result = detection_data[3]
        video_path = detection_data[4]

        # Remove the './static/' prefix from the video path
        if video_path.startswith('./static/'):
            video_path = video_path[len('./static/'):]

        # Replace double backslashes with single ones
        video_path = video_path.replace('\\', '/')

        return render_template('resultVideo.html', user_name=user_name, user_logged_in=user_logged_in,
                               real_confidence=real_confidence, fake_confidence=fake_confidence,
                               overall_result=overall_result, video_path=video_path)
    else:
        return render_template('resultVideo.html')

@app.route('/detectImage', methods=['POST'])
def detect():
    if 'input_image' not in request.files:
        return "No file part"
    
    file = request.files['input_image']
    
    if file.filename == '':
        return "No selected file"
    
    input_image = Image.open(file).convert("RGB")
    
    # Run prediction
    confidences, face_with_mask = predict(input_image)
    
    # Convert confidences to percentage
    real_confidence = round(confidences['real'] * 100, 2)
    fake_confidence = round(confidences['fake'] * 100, 2)
    
    # Determine class based on threshold (e.g., 50%)
    threshold = 50
    if real_confidence >= threshold:
        prediction_class = "real"
    else:
        prediction_class = "fake"
    
    # Convert the PIL image to base64 for display
    _, img_buf_arr = cv2.imencode('.png', face_with_mask)
    img_str = base64.b64encode(img_buf_arr).decode('utf-8')

    image_path = save_image_locally(img_str)

    # Save detection result to database
    if 'id' in session:
        user_id = session.get('id', '')  # Get the user ID from the session
        save_to_database_image(user_id, real_confidence, fake_confidence, prediction_class, image_path, datetime.now())

    # Prepare response
    response = {
        "real_confidence": real_confidence,
        "fake_confidence": fake_confidence,
        "prediction_class": prediction_class,
        "image": img_str
    }
    
    return response

def save_image_locally(img_str):
    # Decode the base64 image string
    img_data = base64.b64decode(img_str)
    
    # Define the directory to save the images
    save_dir = os.path.abspath('static/resultimages')
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate a unique filename based on timestamp and random string
    unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}.png"
    image_path = os.path.join(save_dir, unique_filename)
    
    # Save the image data to the local filesystem
    with open(image_path, 'wb') as f:
        f.write(img_data)

    # Return the relative path to the image
    return f"resultimages/{unique_filename}"

def save_to_database_image(user_id, real_confidence, fake_confidence, prediction_class, image_path, time):
    cursor = mysql.connection.cursor()
    cursor.execute('INSERT INTO detection (user_id, real_confidence, fake_confidence, prediction_class, image_path, time) VALUES (%s, %s, %s, %s, %s, %s)',
                   (user_id, real_confidence, fake_confidence, prediction_class, image_path, time))
    mysql.connection.commit()

@app.route('/history')
def history():
    if 'loggedin' in session and session['loggedin']:
        user_name = session.get('name', '')
        user_logged_in = session.get('loggedin', False)
        user_id = session.get('id', '')  # Get the user ID from the session

        cursor = mysql.connection.cursor()
        
        # Fetch data from the detection table for the current user
        cursor.execute("SELECT * FROM detection WHERE user_id = %s", (user_id,))
        image_results = cursor.fetchall()
        
        # Convert image results to list of dictionaries
        image_detection_history = []
        for row in image_results:
            detection = {
                'id': row[0],
                'real_confidence': row[1],
                'fake_confidence': row[2],
                'prediction_class': row[3],
                'image_path': row[4],
                'time': row[5].strftime('%Y-%m-%d %H:%M:%S')  # Format datetime
            }
            image_detection_history.append(detection)
        
        # Fetch data from the detectionvideo table for the current user
        cursor.execute("SELECT * FROM detectionvideo WHERE user_id = %s", (user_id,))
        video_results = cursor.fetchall()
        
        # Convert video results to list of dictionaries
        video_detection_history = []
        for row in video_results:
            detection = {
                'id': row[0],
                'real_confidence': row[1],
                'fake_confidence': row[2],
                'overall_result': row[3],
                'video_path': row[4],
                'time': row[5].strftime('%Y-%m-%d %H:%M:%S')  # Format datetime
            }
            video_detection_history.append(detection)

        return render_template('history.html', user_name=user_name, user_logged_in=user_logged_in,
                               image_detection_history=image_detection_history, video_detection_history=video_detection_history)
    else:
        return render_template('history.html')

@app.route('/removeDetection/<int:detection_id>', methods=['POST'])
def remove_detection(detection_id):
    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM detection WHERE id = %s", (detection_id,))
    mysql.connection.commit()
    return jsonify({"message": "Detection removed successfully"})

@app.route('/removeDetectionVideo/<int:detection_id>', methods=['POST'])
def remove_detection_video(detection_id):
    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM detectionvideo WHERE id = %s", (detection_id,))
    mysql.connection.commit()
    return jsonify({"message": "Detection removed successfully"})

def predict(input_image:Image.Image):
    """Predict the label of the input_image"""
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')
    face = face.unsqueeze(0) # add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    
    # convert the face into a numpy array to be able to plot it
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    target_layers=[model.block8.branch1[-1]]
    use_cuda = True if torch.cuda.is_available() else False
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"
        
        real_prediction = 1 - output.item()
        fake_prediction = output.item()

        confidences = {
            'real': real_prediction,
            'fake': fake_prediction
        }
    return confidences, face_with_mask

#########################################################video
def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output

@app.route('/detectVideo', methods=['POST'])
def test_full_image_network(video_path='./static/uploadedvideos', model_path='./pretrained_model/deepfake_c0_xception.pkl', output_path='./static/resultvideos',
                            start_frame=0, end_frame=None, cuda=True):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    if 'input_video' not in request.files:
        return "No file part"

    # Get the uploaded file
    uploaded_video = request.files['input_video']

    # If the user does not select a file, the browser submits an empty file without a filename
    if uploaded_video.filename == '':
        return "No selected file"

    # Save the uploaded video to the specified path
    video_filename = secure_filename(uploaded_video.filename)
    video_path = os.path.join(video_path, video_filename)
    uploaded_video.save(video_path)

    print('Starting: {}'.format(video_path))

    reader = cv2.VideoCapture(video_path)

    
    video_fn = video_path.split('/')[-1].split('.')[0]+'.mp4'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')    
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Load model
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if cuda:
        model = model.cuda()

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    real_frames_count = 0
    fake_frames_count = 0

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # Image size
        height, width = image.shape[:2]

        # Init output writer
        if writer is None:
            # Extract the filename and extension from the video path
            video_filename = os.path.basename(video_path)
            video_name, video_ext = os.path.splitext(video_filename)

            # Combine the output directory with the video filename and .mp4 extension
            output_video_path = os.path.join(output_path, video_name + '.mp4')

            # Initialize the output writer
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (height, width)[::-1])


        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]

            # Actual prediction using our model
            prediction, output = predict_with_model(cropped_face, model,
                                                    cuda=cuda)
            # ------------------------------------------------------------------

            # Text and bb
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            label = 'fake' if prediction == 1 else 'real'

            if prediction == 0:
                real_frames_count += 1
            else:
                fake_frames_count += 1

            # Calculate the total number of frames
            total_frames = real_frames_count + fake_frames_count

            # Calculate the percentage of real and fake frames
            percentage_real = (real_frames_count / total_frames) * 100
            percentage_fake = (fake_frames_count / total_frames) * 100

            percentage_real = round((real_frames_count / total_frames) * 100, 2)
            percentage_fake = round((fake_frames_count / total_frames) * 100, 2)
            if(percentage_real>=20):
                dummy = percentage_real
                percentage_real = percentage_fake
                percentage_fake = dummy
                overall_result = "Real"
            else:
                overall_result = "Fake" if fake_frames_count > real_frames_count else "Real"
            # Determine the overall result based on the majority of predictions
            
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            output_list = ['{0:.2f}'.format(float(x)) for x in
                           output.detach().cpu().numpy()[0]]
            cv2.putText(image, str(output_list)+'=>'+label, (x, y+h+30),
                        font_face, font_scale,
                        color, thickness, 2)
            # draw box over face
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        if frame_num >= 50:
            break

        # Show
        cv2.imshow('test', image)
        cv2.waitKey(33)     # About 30 fps
        writer.write(image)
    pbar.close()
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_video_path))

        if 'id' in session:
            user_id = session['id']  # Get the user ID from the session
            save_to_database_video(user_id, real_frames_count, fake_frames_count, overall_result, output_video_path, datetime.now())
        
        return jsonify({
            "output_video_path": output_video_path,
            "overall_result": overall_result,
            "percentage_real": percentage_real,
            "percentage_fake": percentage_fake
        })
    else:
        print('Input video file was empty')
        return "Error: Input video file was empty"

def save_to_database_video(user_id, real_frames_count, fake_frames_count, overall_result, video_path, time):
    cursor = mysql.connection.cursor()
    cursor.execute('INSERT INTO detectionvideo (user_id, real_confidence, fake_confidence, overall_result, video_path, time) VALUES (%s, %s, %s, %s, %s, %s)',
                   (user_id, real_frames_count, fake_frames_count, overall_result, video_path, time))
    mysql.connection.commit()

if __name__ == '__main__':
    app.run(debug=True)