from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
import numpy as np
import glob
import os

class FeaturesExtractor(object):
    def __init__(self, *args, **kwargs):
        self._params = kwargs.get("parameters")
        self._positive_image_path = kwargs.get("pospath")
        self._negative_image_path = kwargs.get("negpath")
        self._positive_features_path = kwargs.get("posfeatpath")
        self._negative_features_path = kwargs.get("negfeatpath")
        self._descriptor_type = kwargs.get("descriptor", "HOG")

        self.__makedir__(self._positive_features_path)
        self.__makedir__(self._negative_features_path)

    def run(self):
        self.__empty_directory__(self._positive_features_path)
        self.__empty_directory__(self._negative_features_path)
        self.__compute__(self._positive_image_path, self._positive_features_path)
        self.__compute__(self._negative_image_path, self._negative_features_path)

    def __empty_directory__(self, dir_path):
        for file_path in glob.glob(os.path.join(dir_path, "*")):
            os.remove(file_path)

    def __makedir__(self, dir_path):
        if not dir_path:
            return

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    def __compute__(self, dir_path, feat_path):
        for im_path in glob.glob(os.path.join(dir_path, "*")):
            im = imread(im_path, as_grey=True)
            fd = self.__descriptor__(im)
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(feat_path, fd_name)
            joblib.dump(fd, fd_path)

    def __descriptor__(self, image):
        switch = {
            "HOG" : hog(
                image,
                self._params["orientations"],
                self._params["pixels_per_cell"],
                self._params["cells_per_block"],
                self._params["visualize"],
                self._params["normalize"] ),
            }
        return switch.get(self._descriptor_type)

