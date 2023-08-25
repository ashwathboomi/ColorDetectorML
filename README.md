## ColorDetectorML
# Created a machine learning model that identifies color schemes within images using TensorFlow

The model and the data can be found in the master branch of the repository.

The data folder contains all the images that were used to train, validate, and test the model. Inside the folder are 6 folders, each representing a color class, which contains images of that color. In total, there are around 1000 images in the data directory.

The models folder contains the fitted model that can be used to determine the color scheme within an image. The input requires any image to be resized to (256, 256) and scaled down. This can be done by the following commands:

```
#import tensorflow, numpy, opencv-python packages
imgtest = cv2.imread('greenredyellowtest.png')
imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)
plt.imshow(imgtest)
resized = tf.image.resize(imgtest, (256,256))
yhat = model.predict(np.expand_dims(resized/255, 0))
```
The logs folder contains the training history of the model

ColorDetector.ipynb contains more imformation on how the data was preprocessed and the model was setup.





