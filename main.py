import cv2
import numpy as np
import tensorflow.keras
from PIL import Image, ImageOps
# Load the cascade face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# Load the cascade eyes
cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray,1.1,6) 
    # Draw rectangle around the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #bottom left and top right coordinate
        crop_img = img[y: y + h, x: x + w] # Crop from x, y, w, h -> 100, 200, 300, 400
        #show and save croped image
        cv2.imshow('imgs',crop_img)
        cv2.imwrite("face.jpg", crop_img)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
    cv2.imshow('img',img)

    ##############################
    np.set_printoptions(suppress=True)
    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5')
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open('face.jpg')
    sizes = (224, 224)
    image = cv2.resize(img, (224,224))
    #image = ImageOps.fit(image, sizes, Image.ANTIALIAS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    prediction = model.predict(data)
    print(prediction)
    ##############################
    k= cv2.waitKey(30) & 0xff
    if k==27:
        break       
cap.release()
cv2.destroyAllWindows()
