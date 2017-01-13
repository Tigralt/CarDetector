# CarDetector
An object detector framework optimized for car detection.
This framework is based on the work of [**bikz05**](https://github.com/bikz05/object-detector.git)

## Basic Usage

```shell
$ git clone https://github.com/tigralt/cardetector.git
$ ./cardetector.py -h
usage: cardetector.py [-h] [-v] [--config PATH] [-e] [-t] [-p IMAGE_PATH]
                      [-f {HOG}] [-c {LIN_SVM}]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity
  --config PATH         The path to the config file
  -e, --extract         Perform feature extraction from training images and
                        save them
  -t, --train           Train and save the classifier
  -p IMAGE_PATH, --predict IMAGE_PATH
                        The path to the image file or dir
  -f {HOG}, --feature {HOG}
                        Select the feature to be extracted
  -c {LIN_SVM}, --classifier {LIN_SVM}
                        Select the classifier to be used
```
### Examples
#### Extract features from images

```
$ ./cardetector.py -e
```

#### Train the classifier

```
$ ./cardetector.py -t
```

#### Predict an image and display it

If you want to predict only one image:
```
$ ./cardetetory -p data/dataset/test/my-image.pgm -v
```

If you want to predict all images from a directory, simply use the directory path then press any key to continue after the image is pressed.
You can also press **q** to quit while the prediction.
```
$ ./cardetetory -p data/dataset/test -v
```

The `-v` parameter is important here, it's thanks to him that the script will display the picture.


### Configuration File

All the configurations files are in the `data/config/` configuration directory. You can add files or change them as per your need. Here is what the default configuration file looks like:

```bash
[HOG]
orientations: 9
pixels_per_cell: [8, 8]
cells_per_block: [3, 3]
visualize: False
normalize: True

[WINDOW]
window_size: [100, 40]
step_size: [10, 10]

[NMS]
threshold: .3

[PATHS]
positive_images_path: data/dataset/car/train/pos
negative_images_path: data/dataset/car/train/neg
positive_features_path: data/features/pos
negative_features_path: data/features/neg
model_path: data/models/svm.model
```

### About the modules

* `classifier.py` -- This module is used to create the classifier needed to train and predict.
* `config.py` -- Imports the configuration variables from `config.cfg`.
* `features.py` -- This module is used to extract features (HOG,...) of the images.
* `nms.py` -- This module performs Non Maxima Suppression.
* `slidingwindow.py` -- This module performs the sliding window.
* `timer.py` -- This modules measure the time elapsed.

## TODO

* Add bootstrapping (Hard Negative Mining).
