# ColorDetectorML
## Created a machine learning model that identifies color schemes within images using TensorFlow

The model and the data can be found in the master branch of the repository.

The data folder contains all the images that were used to train, validate, and test the model. Inside the folder are 6 folders, each representing a color class, which contains images of that color. In total, there are around 1000 images in the data directory. All data was obtained/sourced from Google Images

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

ColorDetector.ipynb contains more information on how the data was preprocessed and the model was set up.

To load the model for personal use, the following script can be used:

```
import os
from tensorflow.keras.models import load_model

use_model = load_model(os.path.join('models', 'ColorDetector.keras'))
```

The output of the model will be a one-hot encoded array of length 6. The distinct color classes are represented by the index position. The following key can be used: \[blue green orange purple red yellow] 

Example: \[1 0 0 1 0 0] indicated that the colors blue and purple are present.

UPDATE 8/25/2023:

The model can identify 6 distinct colors. In identifying the primary color, the model is on average 97% accurate. The accuracy decreases when identifying the secondary and tertiary colors. This is mostly due to the fact that the model was trained on images that contained one color. In the next iterations, a larger image dataset with images containing multiple colors will be used to train the model. This will give the model more credibility in identifying the entire color palette of an image.

In future updates, the model will be implemented in an application that can sort through photos in one's digital album/gallery by color. 






