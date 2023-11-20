import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random
import os
import json
import shutil
import subprocess

start_time = 0
end_time = 0
prev_state = "Forward"
flag = False
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video= str(random.randint(1,50000))+".mp4"
writer= cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height), isColor=True)

def Head_record_duration(text,img):
    global start_time, end_time, recorded_durations, prev_state, flag,writer, video
    outputVideo =''
    if text != "Forward":
        if str(text) != prev_state and prev_state == "Forward":
            start_time = time.time()
            writer.write(img)
        elif str(text) != prev_state and prev_state != "Forward":
            writer.release()
            end_time = time.time()
            duration = math.ceil(end_time - start_time)
            outputVideo = 'HeadViolation' + video
            HeadViolation = {
                "Name": prev_state,
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                "Duration": str(duration) + " seconds",
                "Mark": (2*(duration-3)),
                "Link": outputVideo
            }
            if flag:
                write_json(HeadViolation)
                reduceBitRate(video, outputVideo)
                move_file_to_output_videos(outputVideo)
            os.remove(video)
            start_time = time.time()
            video = str(random.randint(1,50000))+".mp4"
            writer= cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height),isColor=True)
            flag = False
        elif str(text) == prev_state and (time.time() - start_time) > 3:
            flag = True
            writer.write(img)
        elif str(text) == prev_state and (time.time() - start_time) <= 3:
            flag = False
            writer.write(img)
        prev_state = text
    else:
        if prev_state != "Forward":
            writer.release()
            end_time = time.time()
            duration = math.ceil(end_time - start_time)
            outputVideo = 'HeadViolation' + video
            HeadViolation = {
                "Name": prev_state,
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                "Duration": str(duration) + " seconds",
                "Mark": (2 * (duration - 3)),
                "Link": outputVideo
            }
            if flag:
                write_json(HeadViolation)
                reduceBitRate(video, outputVideo)
                move_file_to_output_videos(outputVideo)
            os.remove(video)
            video = str(random.randint(1,50000))+".mp4"
            writer= cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height),isColor=True )
            flag = False
        prev_state = text

# function to add to JSON
def write_json(new_data, filename='violation.json'):
    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data.append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

def move_file_to_output_videos(file_name):
    # Get the current working directory (project folder)
    current_directory = os.getcwd()
    # Define the paths for the source file and destination folder
    source_path = os.path.join(current_directory, file_name)
    destination_path = os.path.join(current_directory, 'OutputVideos', file_name)
    try:
        # Use 'shutil.move' to move the file to the destination folder
        shutil.move(source_path, destination_path)
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found in the project folder.")
    except shutil.Error as e:
        print(f"Error: Failed to move the file. {e}")

def reduceBitRate (input_file,output_file):
   target_bitrate = "1000k"  # Set your desired target bitrate here
   # Specify the full path to the FFmpeg executable
   ffmpeg_path = "C:/Users/kaungmyat/Downloads/ffmpeg-2023-08-28-git-b5273c619d-essentials_build/ffmpeg-2023-08-28-git-b5273c619d-essentials_build/bin/ffmpeg.exe"  # Replace with the actual path to ffmpeg.exe on your system
   # Run FFmpeg command to lower the bitrate
   command = [
      ffmpeg_path,
      "-i", input_file,
      "-b:v", target_bitrate,
      "-c:v", "libx264",
      "-c:a", "aac",
      "-strict", "experimental",
      "-b:a", "192k",
      output_file
   ]
   subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   print("Bitrate conversion completed.")

while cap.isOpened():
    success, image = cap.read()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360

            # print(y)

            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
            elif y > 15:
                text = "Looking Right"
            elif x < -8:
                text = "Looking Down"
            elif x > 15:
                text = "Looking Up"
            else:
                text = "Forward"
            Head_record_duration(text,image)
            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

            cv2.line(image, p1, p2, (255, 0, 0), 2)

            # Add the text on the image
            cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Head Pose Estimation', image)

    # Check if 'finish' key is pressed (q key) or Esc key is clicked
    key = cv2.waitKey(1)
    if key == ord("q") or key == 27:
        break
cv2.destroyAllWindows()
cap.release()
