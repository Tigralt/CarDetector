# CarDetector
An object detector framework optimized for car detection.
This framework is based on the work of [**bikz05**](https://github.com/bikz05/object-detector.git)

## Run the code

```shell
git clone https://github.com/tigralt/cardetector.git
./cardetector.py
```

### Configuration File

All the configurations are in the `data/config/config.cfg` configuration files. You can change it as per your need. Here is what the default configuration file looks like-

```bash
[hog]
min_wdw_sz: [100, 40]
step_size: [10, 10]
orientations: 9
pixels_per_cell: [8, 8]
cells_per_block: [3, 3]
visualize: False
normalize: True

[nms]
threshold: .3

[paths]
pos_feat_ph: ../data/features/pos
neg_feat_ph: ../data/features/neg
model_path: ../data/models/svm.model
```

### About the modules

* `features.py` -- This module is used to extract features (HOG,...) of the images.
* `classifier.py` -- This module is used to create the classifier needed to train and predict.
* `nms.py` -- This module performs Non Maxima Suppression.
* `slidingwindow.py` -- This module performs the sliding window.
* `config.py` -- Imports the configuration variables from `config.cfg`.

## TODO

* Faster NMS code.
* Add bootstrapping (Hard Negative Mining) code.
