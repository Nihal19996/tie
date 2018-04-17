import face_recognition
import cv2
import time
from imutils.video import VideoStream
import imutils
import numpy as np


# Load a sample picture and learn how to recognize it.
# kushal_image = face_recognition.load_image_file("kushal.jpg")
# kushal_face_encoding = face_recognition.face_encodings(kushal_image)[0]
#
# # Load a second sample picture and learn how to recognize it.
# nihal_image = face_recognition.load_image_file("nihal.jpg")
# nihal_face_encoding = face_recognition.face_encodings(nihal_image)[0]
#
# bhavin_image = face_recognition.load_image_file("bhavin.jpg")
# bhavin_face_encoding = face_recognition.face_encodings(bhavin_image)[0]
#
# yash_image = face_recognition.load_image_file("yash.jpg")
# yash_face_encoding = face_recognition.face_encodings(yash_image)[0]
#
#
#     # Create arrays of known face encodings and their names
# known_face_encodings = [
#     kushal_face_encoding,
#     nihal_face_encoding,
#     bhavin_face_encoding,
#     yash_face_encoding
# ]
# known_face_names = [
#     "Kushal",
#     "Nihal",
#     "Bhavin",
#     "Yash"
# ]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
hello_world = 5


class VideoCamera(object):
    def __init__(self):
        self.vs = VideoStream(src="rtsp://admin:12345@103.59.59.10:554/h264/ch1/main/av_stream").start()
        # self.vs = VideoStream(src=0).start()
        time.sleep(2)
        # self.process_this_frame = True
        # self.face_locations = []
        # self.face_encodings = []
        # self.face_names = []
        self.train()

    def __del__(self):
        self.vs.stop()
        # self.video.release()

    def train(self):
        kushal_image = face_recognition.load_image_file("kushal.jpg")
        kushal_face_encoding = face_recognition.face_encodings(kushal_image)[0]

        # Load a second sample picture and learn how to recognize it.
        nihal_image = face_recognition.load_image_file("nihal.jpg")
        nihal_face_encoding = face_recognition.face_encodings(nihal_image)[0]

        bhavin_image = face_recognition.load_image_file("bhavin.jpg")
        bhavin_face_encoding = face_recognition.face_encodings(bhavin_image)[0]

        yash_image = face_recognition.load_image_file("yash.jpg")
        yash_face_encoding = face_recognition.face_encodings(yash_image)[0]

        # Create arrays of known face encodings and their names
        self.known_face_encodings = [
            kushal_face_encoding,
            nihal_face_encoding,
            bhavin_face_encoding,
            yash_face_encoding
        ]

        self.known_face_names = [
            "Kushal",
            "Nihal",
            "Bhavin",
            "Yash"
        ]

    def get_frame(self):
        global face_locations
        global face_encoding
        global face_names
        global process_this_frame

        frame = self.vs.read()
        # frame = imutils.resize(frame, width=400)

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognitiongnition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time

        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # print("{}{}{}{}".format(top, right, bottom, left))
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom + 30), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 35, bottom + 30), font, 1.0, (255, 255, 255), 1)
            # print("found")

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg



