# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.image import encode_jpeg
from imutils.video import VideoStream
import numpy as np 	
from flask import Response
from flask import request
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

from scipy.spatial import distance as dist


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


# load our serialized face detector model from disk
prototxtPath = r"./deploy.prototxt"
weightsPath = r"./res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

face_model = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
# sio = socketio.Client()
# sio.connect('http://localhost:3000')

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

source =None
vs = VideoStream(src=0).start()

@app.route("/")
def index():
	# return the rendered template
	print("##\n")
	return render_template("index.html")


def detect_mask():
  # grab global references to the video stream, output frame, and
  # lock variables
	global vs, outputFrame, lock
  

	while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=1280)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

		# loop over the detected face locations and their corresponding
		# locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		
		face_cor = face_model.detectMultiScale(frame)
		frame = cv2.putText(frame, " "+" Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
		stack_x = []
		stack_y = []
		stack_x_print = []
		stack_y_print = []
		global D

		if len(face_cor) == 0:
			pass
		else:
			for i in range(0,len(face_cor)):
				x1 = face_cor[i][0]
				y1 = face_cor[i][1]
				x2 = face_cor[i][0] + face_cor[i][2]
				y2 = face_cor[i][1] + face_cor[i][3]

				mid_x = int((x1+x2)/2)
				mid_y = int((y1+y2)/2)
				stack_x.append(mid_x)
				stack_y.append(mid_y)
				stack_x_print.append(mid_x)
				stack_y_print.append(mid_y)

				# frame = cv2.circle(frame, (mid_x, mid_y), 3 , [255,0,0] , -1)
				# frame = cv2.rectangle(frame , (x1, y1) , (x2,y2) , [0,255,0] , 2)
			
			if len(face_cor) == 2:
				D = int(dist.euclidean((stack_x.pop(), stack_y.pop()), (stack_x.pop(), stack_y.pop())))
				frame = cv2.line(frame, (stack_x_print.pop(), stack_y_print.pop()), (stack_x_print.pop(), stack_y_print.pop()), [0,0,255], 2)
			else:
				D = 0

			if D<250 and D!=0:
				frame = cv2.putText(frame, "!!MOVE AWAY!!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,2, [0,0,255] , 4)
				frame = cv2.putText(frame, str(D/10) + " cm", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 2, cv2.LINE_AA)

    # show the output frame
    # cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		with lock:
			outputFrame = frame.copy()

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break


def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	global source, vs
	# source=request.args.get('source')
	# if source=='0':
	# 	source=0
	# vs = VideoStream(src=source).start()
	print("####FEED####", source)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_mask, args=())
	t.daemon = True
	t.start()
	# start the flask app

	app.run(host='localhost', port='8000', debug=True, threaded=True, use_reloader=False)


# release the video stream pointer
vs.stop()


