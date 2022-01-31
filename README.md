# Sign-Language-Recognition-Using-LSTM

## General

After the start of the pandemic the use of E-Learning has increased drastically. By 2026 the E-learning industry is seat to reach at least $400 billion. With the rise of this new fashion of education no child should be left behind. Children with the disability of hearing loss are held at a disadvantage due to the reason that they have a hard time hearing the lectures that are present on the screen. Other factors would be a mute child who would not be able to speak in according to the response of the lecturer. In order for these children to cope with education the American Sign Language was created not only to help them with education but to provide ease for their daily lives. To help these children with their education we proposed a model that will help the student make ASL gestures to the camera and have it read and provide feedback on what language was read. For this we used OpenCV with Mediapipe Holistic to identify the key markers of the poser with all the values to be collected and then be trained on the Long Short Term Memory Architecture. 

## MediaPipe Holistic
The initial step of the program is to identify the Keypoints of the user. MP Holistic is an API created by Mediapipe which identifies the keypoints on a body by identifying the holistic key points

`mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils`

These two api calls, are calls the Holistic API uses to identify the keypoints holistic keypoints and draw it



The function
`draw_landmarks(image, results)`

Draws landmarks of the identified keypoints on the user

![Unstyled](https://github.com/evarghese563/Images/blob/main/Sign/unstyled.gif)


However wth a few extra lines of code 
`draw_styled_landmarks(image,results)`

The keypoints can be color coded for the viewer to identify the different parts of the human

![Styled](https://github.com/evarghese563/Images/blob/main/Sign/styled.gif)


## Collecting the Data
After the mediapipe has been defined the keypoints must be collected and stored to be trained on later. 

`def extract_keypoints(results)` 

This takes in the extracted keypoints from the video and concatenated the values into a numpy array.


The collected data is abeled as 'Hello', 'Thanks', and 'I Love You'. In order to collect the data the user poses for the three different action in for 30 sequences for 30 frames. For a total of 90 collectd sequences

The identified keypoints are stored in the folder called MP_Data, with each sequence in its respective action folder in the .npy format 


## Training the Long Short Term Memory
To train the Long Short Term Memory the model selected was a Sequential model trained on 3 layers and 2 dense layers. Since it is catagorical data the model was compiled as:

`model.compile(optimizer = 'Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])`

The model was then Trained on 2000 epochs.

## Testing in Real time
These are the testing results in real time from the project after training the data on the LSTM Architecture using MP Holistic

### Hello

![Hello](https://github.com/evarghese563/Images/blob/main/Sign/hello.gif)

### Thanks

![Thanks](https://github.com/evarghese563/Images/blob/main/Sign/thanks.gif)

### I Love You

![ILY](https://github.com/evarghese563/Images/blob/main/Sign/ily.gif)

### Extras

#### 1.  Color Coded Probability Viewer

- This is a probability bar which shows the gesture that is being identified the most in the test. The bar rises with the gesture it recognizes most. The color for each bar is as follows:

  - Blue/Hello
  - Green/Thanks
  - Orange/I Love You


## Run
### Run on Old Weights
<!-- To run on the previous versions weights the user has to simply run 'Run.py' or ' RunNoMarker.py'. Make sure either one of the files in the same folder as MP_DATA and actions.h5
 -->
To run on the previous versions weights the user has to simply run 'Run.py' or ' RunNoMarker.py'. Make sure either one of the files in the same folder with actions.h5

### Run on New Weights
To run on a new dataset:

1. CollectData.py

      - When the camera opens pose in the Sign Language poses for 30 sequences in 'Hello', 'Thanks', and 'I Love You'.

2. NeuralNetwork.py

      - To train the collected data on the Long Short Term Memory Architecture hit run

3. Run.py or RunNoMarker.py 

      - Test the model in real time
      - Run RunNoMarker.py if the user would like to run the model without any of the viewing markers the user can run this file to access the application without any of the landmarks

