from ..utils.statics import Timer


class RGBDetector(object):
    """Base class for first stage: 2D RGB detection."""

    def __init__(self, detector_cfg=None):
        self.cfg = detector_cfg
        self.model = None
        self.ifshow = False
        self.ifverbose = False
        self.iftimecnt = False
        self.timestep = 3
        self.timer = Timer(self.timestep, ["Preprocess", "RGB Detection", "Postprocess"])

    def verbose(self, choose:bool):
        """
        choose if output detection result to command window.
        """
        self.ifverbose = choose

    def show(self, choose: bool):
        """
        choose if visualize detection result.
        """
        self.ifshow = choose

    def timecnt(self, choose: bool):
        """
        choose if visualize detection result.
        """
        self.iftimecnt = choose

    def preprocess(self, img):
        """
        Preprocess the images.

        :param img: A single image.

        :return: np.ndarray: A processed image can be fed to model.
        """
        pass

    def build_model(self, cfg):
        """Build model and load weights."""
        pass

    def detect(self, img):
        """Get 2D RGB Detection results. And visualize the detections if set to show.

        Args:
            img (np.ndarray): A single image.

        Returns:
            tuple: Detection results, including bboxes and/or instance masks.
            """
        if self.iftimecnt:
            self.timer.start()
        p_img = self.preprocess(img)
        if self.iftimecnt:
            self.timer.count()
        output = self.model(p_img)[0]
        if self.iftimecnt:
            self.timer.count()
        result = self.postprocess(output, self.cfg.post)
        if self.iftimecnt:
            self.timer.count()
        if self.ifshow:
            self.show_result(img, result, self.cfg.show)
        if self.ifverbose and self.timecnt:
            self.timer.show()
        return result

    def show_result(self, img, result, show_cfg=None):
        """Visualize the detection result"""
        pass

    def postprocess(self, output, post_cfg=None):
        """
        Necessary process on result the model outputs.

        :param result: model outputs
        :return: Detection results, including bboxes and/or instance masks.
        """
        pass