from typing import List
import vart
import cv2
import numpy as np
import utils
from pynq_dpu import DpuOverlay


class FaceDetection:
    def __init__(self, det_threshold=0.55, nms_threshold=0.35, model_path="./faceDetection.xmodel"):
        """
        Args:
            det_threshold: float number for the detection threshold
            nms_threshold: float number for the nms threshold
        """
        self.__overlay = DpuOverlay("dpu.bit")
        self.__overlay.load_model(model_path)
        self.__faceDetection: vart.Runner = self.__overlay.runner
        self.__inputTensors = self.__faceDetection.get_input_tensors()
        self.__outputTensors = self.__faceDetection.get_output_tensors()

        self.__shapeIn: tuple = tuple(self.__inputTensors[0].dims)
        self.__shapeOut: tuple = tuple(self.__outputTensors[0].dims)
        self.__shapeOut2: tuple = tuple(self.__outputTensors[1].dims)
        self.__outputSize: int = int(
            self.__outputTensors[0].get_data_size() / self.__shapeIn[0]
        )
        self.__outputSize2: int = int(
            self.__outputTensors[1].get_data_size() / self.__shapeIn[0]
        )
        self.__output_data: list[np.ndarray] = [
            np.empty(self.__shapeOut, dtype=np.float32, order="C"),
            np.empty(self.__shapeOut2, dtype=np.float32, order="C"),
        ]
        self.__input_data: list[np.ndarray] = [
            np.empty(self.__shapeIn, dtype=np.float32, order="C")
        ]
        self.__image: np.ndarray = self.__input_data[0]
        self.__means: list[int] = [128, 128, 128]
        self.__input_width: int = 320
        self.__input_height: int = 320

        self.__detThreshold: float = det_threshold
        self.__nmsThreshold: float = nms_threshold

    def __mean_image_subtraction(self, image) -> np.ndarray:
        """
        Args:
            image: cv2 image

        Returns: new image with means subtracted
        """
        B, G, R = cv2.split(image)
        B = B - self.__means[0]
        G = G - self.__means[1]
        R = R - self.__means[2]
        image = cv2.merge([B, G, R])
        return image

    def __preprocess(self, image) -> np.ndarray:
        """
        Args:
            image: cv image

        Returns:
            image: preprocessed image
        """
        image: np.ndarray = self.__mean_image_subtraction(image)
        image = cv2.resize(image, (self.__input_width, self.__input_height))
        return image

    def __postprocess(self, img_width, img_height) -> List[int]:
        """

        Args:
            img_width (int): image width
            img_height (int): image height

        Returns: the faces' bounding boxes found in the image

        """
        scale_h: float = img_height / self.__input_height
        scale_w: float = img_width / self.__input_width

        output_data_0: np.ndarray = self.__output_data[0].reshape(1, self.__outputSize)
        bboxes: np.ndarray = np.reshape(output_data_0, (-1, 4))
        output_data_1: np.ndarray = self.__output_data[1].reshape(1, self.__outputSize2)
        scores: np.ndarray = np.reshape(output_data_1, (-1, 2))

        gy = np.arange(0, self.__outputTensors[0].dims[2])
        gx = np.arange(0, self.__outputTensors[0].dims[1])
        [x, y] = np.meshgrid(gx, gy)
        x = x.ravel() * 4
        y = y.ravel() * 4
        bboxes[:, 0] = bboxes[:, 0] + x
        bboxes[:, 1] = bboxes[:, 1] + y
        bboxes[:, 2] = bboxes[:, 2] + x
        bboxes[:, 3] = bboxes[:, 3] + y

        softmax = utils.softmax(scores)

        """ Keep the faces above detection threshold """
        prob = softmax[:, 1]
        keep_idx = prob.ravel() > self.__detThreshold
        bboxes = bboxes[keep_idx, :]
        bboxes = np.array(bboxes, dtype=np.float32)
        prob = prob[keep_idx]

        """ Perform Non-Maxima Suppression """
        face_indices = []
        if len(bboxes) > 0:
            face_indices = utils.nms_boxes(bboxes, prob, self.__nmsThreshold)

        faces = bboxes[face_indices]

        # extract bounding box for each face
        for i, face in enumerate(faces):
            xmin: int = int(max(face[0] * scale_w, 0))
            ymin: int = int(max(face[1] * scale_h, 0))
            xmax: int = int(min(face[2] * scale_w, img_width - 1))
            ymax: int = int(min(face[3] * scale_h, img_height - 1))
            faces[i] = (xmin, ymin, xmax, ymax)

        return faces

    def run(self, image) -> List[int]:
        """

        Args:
            image (np.ndarray): cv image to run the AI task
        Returns:
            AI task output
        """
        preprocessed_image = self.__preprocess(image=image)
        self.__image[0, ...] = preprocessed_image.reshape(self.__shapeIn[1:])
        # execute on DPU
        job_id = self.__faceDetection.execute_async(
            self.__input_data, self.__output_data
        )
        self.__faceDetection.wait(job_id)
        return self.__postprocess(image.shape[1], image.shape[0])
