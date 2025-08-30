# !pip install opencv-python mtcnn Pillow
#does same work as extrtactFrames_detectFaces.py but with a different approach and structure..
import os
import cv2
from mtcnn import MTCNN
from PIL import Image
from tqdm import tqdm

# Set paths
video_dir_real = 'C:\\Users\\shiva\\OneDrive\\Desktop\\DeepFake Detection(2)\\dataset\\processed\\real'
video_dir_fake = 'C:\\Users\\shiva\\OneDrive\\Desktop\\DeepFake Detection(2)\\dataset\\processed\\fake'
output_dir_real = 'C:\\Users\\shiva\\OneDrive\\Desktop\\DeepFake Detection(2)\\dataset\\processed\\fake'
output_dir_fake = 'dataset/processed/fake'

# Ensure output directories exist
os.makedirs(output_dir_real, exist_ok=True)
os.makedirs(output_dir_fake, exist_ok=True)

# Face detector
detector = MTCNN()

def extract_faces(video_path, output_path, max_frames=50):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        faces = detector.detect_faces(frame)
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
            img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            img = img.resize((299, 299))
            img.save(os.path.join(output_path, f"{os.path.basename(video_path)}_f{count}_{i}.jpg"))
            saved += 1
    cap.release()
    return saved

# Run on all videos
def preprocess_all(video_dir, output_dir):
    for file in tqdm(os.listdir(video_dir)):
        if file.endswith(".mp4"):
            video_path = os.path.join(video_dir, file)
            extract_faces(video_path, output_dir)

# Run preprocessing
preprocess_all(video_dir_real, output_dir_real)
preprocess_all(video_dir_fake, output_dir_fake)
# end of file
# This script processes videos from two directories (real and fake), extracts faces using MTCNN, and saves them in specified output directories.    
# The script uses OpenCV for video processing and MTCNN for face detection, ensuring that the output directories exist before saving the images.
# The script is designed to handle a maximum of 50 frames per video and saves the detected faces as images in the output directory. 
# The images are resized to 299x299 pixels, which is suitable for input into the Xception model.
# The script uses the tqdm library to provide a progress bar for the video processing, making it easier to track the progress of the script.
