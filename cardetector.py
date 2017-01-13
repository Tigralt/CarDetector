#! /usr/bin/env python3

from cardetector.config import Config
from cardetector.classifier import Classifier
from cardetector.features import FeaturesExtractor
from cardetector.slidingwindow import SlidingWindow
from cardetector.timer import Timer
import argparse
import glob
import sys
import cv2
import os

FEATURES = ["HOG"]
CLASSIFIERS = ["LIN_SVM"]

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("--config", help="The path to the config file", metavar="PATH", default="data/config/config.cfg")
    parser.add_argument("-e", "--extract", help="Perform feature extraction from training images and save them", action="store_true")
    parser.add_argument("-t", "--train", help="Train and save the classifier", action="store_true")
    parser.add_argument("-p", "--predict", metavar="IMAGE_PATH", help="The path to the image file or dir")
    parser.add_argument("-f", "--feature", help="Select the feature to be extracted", default=FEATURES[0], choices=FEATURES)
    parser.add_argument("-c", "--classifier", help="Select the classifier to be used", default=CLASSIFIERS[0], choices=CLASSIFIERS)
    args = parser.parse_args()

    # Load config
    CONFIG = Config(args.config)
    TIMER = Timer()

    ef = FeaturesExtractor( pospath=CONFIG["PATHS"]["positive_images_path"],
                            negpath=CONFIG["PATHS"]["negative_images_path"],
                            posfeatpath=CONFIG["PATHS"]["positive_features_path"],
                            negfeatpath=CONFIG["PATHS"]["negative_features_path"],
                            parameters=CONFIG[args.feature]
                            )
        
    c = Classifier( posfeatpath=CONFIG["PATHS"]["positive_features_path"],
                    negfeatpath=CONFIG["PATHS"]["negative_features_path"],
                    classifier=args.classifier,
                    model_path=CONFIG["PATHS"]["model_path"]
                    )
        
    if args.extract:
        TIMER.start()
        ef.run()
        TIMER.stop()

        if args.verbose:
            TIMER.print("Extracting features took {} ms.")

    if args.train:
        TIMER.start()
        c.train()
        TIMER.stop()

        if args.verbose:
            TIMER.print("Training took {} ms.")

    if args.predict:
        sw = SlidingWindow( CONFIG["WINDOW"]["window_size"],
                            CONFIG["WINDOW"]["step_size"] )

        if os.path.isdir(args.predict):
            images = glob.glob(os.path.join(args.predict, "*"))
        elif os.path.isfile(args.predict):
            images = [args.predict]
        else:
            sys.exit("[ERROR] The path to predict is neither a dir nor a file!")

        for image in images:
            TIMER.start()
            result = c.predict(image, ef, sliding_window=sw)
            TIMER.stop()
            
            if args.verbose:
                TIMER.print("Prediction took {} ms.")

                # DISPLAY
                im = cv2.imread(image)
                for (x_tl, y_tl, _, w, h) in result:
                    cv2.rectangle(im, (int(x_tl), int(y_tl)), (int(x_tl+w), int(y_tl+h)), (0, 0, 255), thickness=2)
                cv2.imshow("Detection", im)

                if cv2.waitKey() & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
                else:
                    cv2.destroyAllWindows()
