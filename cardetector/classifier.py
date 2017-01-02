# Import the required modules
from skimage.transform import pyramid_gaussian
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
from skimage.io import imread
from cardetector.slidingwindow import SlidingWindow
from cardetector.nms import nms
import numpy as np
import glob
import os

class Classifier(object):
    _fds = []
    _labels = []
    _clf = None

    def __init__(self, **kwargs):
        self._positive_features_path = kwargs.get("posfeatpath")
        self._negative_features_path = kwargs.get("negfeatpath")
        self._classifier = kwargs.get("classifier")
        self._model_path = kwargs.get("model_path")

    def __load_features__(self, path, response):
        for feat_path in glob.glob(os.path.join(path,"*.feat")):
            fd = joblib.load(feat_path)
            self._fds.append(fd)
            self._labels.append(response)

    def __load__(self, model_path):
        self._clf = joblib.load(model_path)

    def __train__(self):
        if self._classifier == "LIN_SVM":
            self._clf = LinearSVC()
            self._clf.fit(self._fds, self._labels)

    def __save__(self, model_path):
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        joblib.dump(self._clf, model_path)

    def train(self):
        self.__load_features__(self._positive_features_path, 1)
        self.__load_features__(self._negative_features_path, 0)
        self.__train__()
        self.__save__(self._model_path)

    def predict(self, image_path, feature_extractor, **kwargs):
        image = imread(image_path, as_grey=False)
        scale = 0
        sliding_window = kwargs.get("sliding_window", SlidingWindow(image.shape, (0,0)))
        downscale = kwargs.get("downscale", 1.25)
        threshold = kwargs.get("threshold", .3)
        
        self.__load__(self._model_path)
        
        detections = []
        min_wdw_sz = sliding_window.getWindowSize()

        for im_scaled in pyramid_gaussian(image, downscale=downscale):
            cd = []
            if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
                break
 
            for (x, y, im_window) in sliding_window.compute(im_scaled):
                if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                    continue

                fd = feature_extractor.__descriptor__(im_window).reshape(1, -1)
                pred = self._clf.predict(fd)
                if pred == 1:
                    detections.append((x, y, self._clf.decision_function(fd),
                        int(min_wdw_sz[0]*(downscale**scale)),
                        int(min_wdw_sz[1]*(downscale**scale))))
                    cd.append(detections[-1])
            scale += 1
        
        detections = nms(np.array(detections), threshold)
        return detections
        



if __name__ == "__main__":
    # Example
    p = "../data/features/pos"
    n = "../data/features/neg"
    m = "../data/models/svm.model"
    i = "../data/dataset/CarData/TestImages/test-16.pgm"
    c = Classifier(posfeatpath=p, negfeatpath=n, classifier="LIN_SVM", model_path=m)

    c.train()
