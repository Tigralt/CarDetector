class SlidingWindow(object):
    def __init__(self, window_size, step_size):
       self._window_size = window_size
       self._step_size = step_size

    def setWindowSize(self, window_size):
        self._window_size = window_size

    def getWindowSize(self):
        return self._window_size

    def setStepSize(self, step_size):
        self._step_size = step_size

    def getStepSize(self):
        return self._step_size
    
    def compute(self, image):
        '''
        This function returns a patch of the input image `image` of size equal
        to `window_size`. The first image returned top-left co-ordinates (0, 0) 
        and are increment in both x and y directions by the `step_size` supplied.
        So, the input parameters are -
        * `image` - Input Image
        * `window_size` - Size of Sliding Window
        * `step_size` - Incremented Size of Window

        The function returns a tuple -
        (x, y, im_window)
        where
        * x is the top-left x co-ordinate
        * y is the top-left y co-ordinate
        * im_window is the sliding window image
        '''
        for y in range(0, image.shape[0], self._step_size[1]):
            for x in range(0, image.shape[1], self._step_size[0]):
                yield (x, y, image[y:(y + self._window_size[1]), x:(x + self._window_size[0])])

