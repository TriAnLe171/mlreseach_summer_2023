# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import os
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from contrast import increase_contrast

image_og=cv2.imread('ex2.png')
image_og = imutils.resize(image_og, height=500)

image=increase_contrast(image_og)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

cv2.imshow('og',image_og)
cv2.waitKey(0)
cv2.destroyAllWindows()


# find contours in the edge map, then sort them by their
# size in descending order
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)


## contours of the display
displayCnt = None
# # loop over the contours
for c in cnts:
	# approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if the contour has four vertices, then we have found
    # the thermostat display
    if len(approx) == 4:
        displayCnt = approx
    # break
    
# extract the display, apply a perspective transform
# to it
if displayCnt is not None:
    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    output = four_point_transform(image_og, displayCnt.reshape(4, 2))

    thresh = cv2.threshold(warped, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #   crop the image to avoid thick display edges
    thresh=thresh[15:thresh.shape[0]-15,15:thresh.shape[1]-15]
    # thresh = increase_contrast(thresh)
    cv2.imshow('thresh',thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
     print("No display contours found")

     
# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []

# loop over the digit area candidates
xy=[]
for dgit in cnts:
	# compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(dgit)
	# if the contour is sufficiently large, it must be a digit
    if w >= 10 and h >= 10:
        xy.append((x,y,w,h))
        digitCnts.append(dgit)

print(len(cnts))
print(len(digitCnts))
print(len(xy))
        

#Draw bounding boxes around the digits
if len(digitCnts)>0:
    output_copy=output.copy()
    for (x, y, w, h) in xy:
        # cv2.rectangle(output1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.rectangle(output_copy, (x+15, y+15), (x + w+15, y + h+15), (0, 255, 0), 2)

else:
    print('NO')

cv2.imshow('Input',image)
cv2.imshow('Output',output_copy)
cv2.waitKey(0)
cv2.destroyAllWindows


#Extract and save the images of dgits to a folder for further training
digit_images = "digit_images"
for i, (x, y, w, h) in enumerate(xy):

    # Crop the digit region from the image and invert the color
    digit_roi = cv2.bitwise_not(output[y:y+h+30, x:x+w+30])
    # Save the digit as an individual image
    digit_path = os.path.join(digit_images, f"digit_{i}.png")
    cv2.imwrite(digit_path, digit_roi)

# Load MNIST dataset
(train_images, train_labels), (test_images_mnist, test_labels_mnist) = tf.keras.datasets.mnist.load_data()

# Convert images and labels to tensors
train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int32)

# Create a TensorFlow dataset from the data
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

class_names = ['0','1','2','3','4','5','6','7','8','9']
# Shuffle and batch the training dataset
batch_size = 64
train_dataset = train_dataset.shuffle(buffer_size=len(train_images)).batch(batch_size)

# Choose CPU or GPU to execute the code
devices = tf.config.list_physical_devices("GPU")
if len(devices) > 0:
    device = "GPU"
else:
    device = "CPU"

print(f"Using {device} device")

# Convolution Neural Network
# DefaultConv2D=partial(tf.keras.layers.Conv2D, kernel_size=3, activation = 'relu', padding= 'SAME')
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu', input_shape= (28,28,1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='softmax'))
model.add(tf.keras.layers.Dense(10))
model.summary()

test_folder= 'D:\mlr\digit_detection\digit_images'
test_images=[]
test_labels=[5,0,4,1,9,2,1,3,1,4,3,5,3,6,1,7,2,8,6,9,4,0,9,1,1]
#iterate through .png files
# Iterate through .png files
for filename in os.listdir(test_folder):
    if filename.endswith(".png"):
        # Read the image and convert it to grayscale
        image_path = os.path.join(test_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))  # Resize to desired size
        image = image.astype(np.float32) / 255.0 
        # Append the image and label to the test datasets
        test_images.append(image)

# Convert the lists to NumPy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Convert images and labels to tensors
test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int32)

# Combine MNIST testing dataset with custom testing dataset
test_images_combined = tf.concat([test_images_mnist, test_images], axis=0)
test_labels_combined = tf.concat([test_labels_mnist, test_labels], axis=0)

# Create test dataset
test_dataset_combined = tf.data.Dataset.from_tensor_slices((test_images_combined, test_labels_combined))
test_dataset_batched = test_dataset_combined.batch(batch_size)

print(train_images.shape)
print(train_labels.shape)
print(test_images_combined.shape)
print(test_labels_combined.shape)

num_epochs=2

# model.compile(
#             optimizer=tf.keras.optimizers.Adam(0.0003),
#             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#             metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], #the model will compute the accuracy using SparseCategoricalAccuracy()
#         )
#         # Training process
# model.fit(
#             train_dataset,
#             epochs=num_epochs,
#             batch_size=batch_size,
#         )

# test_loss, test_accuracy = model.evaluate(test_dataset_batched)
# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy}")

#Perform prediciton on i-th image
# i=10024

# img=test_images_combined[i]
# img = tf.expand_dims(img, axis=0)

# # Perform prediction on the digit image
# prediction = model.predict(img)
# predicted_label = np.argmax(prediction)
# confidence = np.max(prediction)

# print(confidence)
# print(predicted_label)


# Perform prediction on the image using the trained model
for i, (filename, (x, y, w, h)) in enumerate(zip(os.listdir(test_folder), xy)):
    if filename.endswith(".png"):
        # Load the image and preprocess it
        image_path = os.path.join(test_folder, filename)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, (28, 28))  # Resize to (28, 28)
        img = img.reshape((1,28, 28,1))  # Reshape to (1, 28, 28, 1)
        img = img.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
        
        # update the weights after each prediction
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], #the model will compute the accuracy using SparseCategoricalAccuracy()
        )
        # Training process
        model.fit(
            train_dataset,
            epochs=num_epochs,
            batch_size=batch_size,
        )

        # Test our own dataset
        test_loss, test_accuracy = model.evaluate(test_dataset_batched)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

        # Perform prediction on the digit image
        prediction = model.predict(img)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)

        # Prepare the text to display
        label_text = f"Label: {predicted_label}"
        accuracy_text = f"Accuracy: {confidence:.2f}"

        # Draw the text above the bounding box
        cv2.putText(output, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),thickness=1)
        cv2.putText(output, accuracy_text, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),thickness=1)

scale_percent = 120  # Percentage by which you want to scale the image
width = int(output.shape[1] * scale_percent / 100)
height = int(output.shape[0] * scale_percent / 100)
output_resized = cv2.resize(output, (width, height))
# Display the image with bounding boxes and predictions
cv2.imshow("Image with Predictions", output_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
