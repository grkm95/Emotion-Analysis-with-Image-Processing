import cv2
import sys
from keras.models import load_model
import numpy as np
sys.path.insert(0, '/home/gorkem/PycharmProjects/image_processing')

faceCascade = cv2.CascadeClassifier("/home/gorkem/PycharmProjects/emotion/haarcascade_frontalface_default.xml")

emotion_dict = {0: "KIZGIN", 1: "TIKSINME", 2: "KORKU", 3: "MUTLU", 4: "UZGUN", 5: "SASIRMIS", 6: "NORMAL"}

MODELPATH = '/home/gorkem/PycharmProjects/image_processing/models/model_new.h5'

model = load_model(MODELPATH)

video_capture = cv2.VideoCapture(0)

while True:

    # kare-kare yakalama
    retval, frame = video_capture.read()

    # griye dönüştürme
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Haar Cascade ile algılama
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35)
    )

    # algılanan yüzün kareyle tanımlanması
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        prediction = model.predict(cropped_img)
        cv2.putText(frame, emotion_dict[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                    1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
