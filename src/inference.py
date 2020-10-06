import tensorflow as tf
import coloredlogs
import argparse
import cv2
import numpy as np
import tensorflow.keras.backend as K
import pandas as pd
import json

from src.util import draw_boxes_on_image

logger = tf.get_logger()
coloredlogs.install(
    logger=logger,
    level=tf.compat.v1.logging.INFO,
    fmt="%(asctime)s ModelTraining] [%(levelname)s] %(message)s"
)
logger.propagate = False


def parse_args():
    parser = argparse.ArgumentParser(description="Creating tfrecord dataset")
    parser.add_argument("--image_path", required=True, help="Path to test image")
    parser.add_argument("--model_dir", required=True, help="Path to trained model")
    parser.add_argument("--label_mapping", help="Path to label mapping file")
    parser.add_argument("--image_annotation_file", help="Path to file containing image annotations")
    return vars(parser.parse_args())


class TomatoClassifier:
    def __init__(self, model_path):
        self._model = tf.keras.models.load_model(model_path)
        self._input_size = self._model.get_layer(index=0).output_shape[0][1:3]
        self._last_conv_layer = self._model.get_layer(index=-3)
        self._weights = self._model.get_layer(index=-1).get_weights()[0]
        self._get_activations = K.function([self._model.layers[2].input], [self._last_conv_layer.output])

    def has_tomatoes(self, image_path, threshold=0.5):
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=self._input_size)
        image = tf.keras.preprocessing.image.img_to_array(image)
        logit = self._model.predict(image[tf.newaxis, ...])[0][0]
        probability = tf.nn.sigmoid(logit).numpy()

        return probability >= threshold

    def get_cam(self, image_path):
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=self._input_size)
        image = tf.keras.preprocessing.image.img_to_array(image)

        [activations] = self._get_activations(image[tf.newaxis, ...])
        activations = activations[0, :, :, :]

        # Generate classification activation map
        cam = np.zeros(dtype=np.float32, shape=activations.shape[:2])
        for i, w in enumerate(self._weights[:, 0]):
            cam += w * activations[:, :, i]

        # Normalize CAM
        cam /= np.max(cam)

        return cam


def visualize_prediction(image_path, prediction, cam, annotations=None, label_mapping=None):
    if prediction:
        message = "Tomato traces: Yes"
    else:
        message = "Tomato traces: No"
    image = cv2.imread(image_path)
    cam = cv2.resize(cam, image.shape[:2])
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_TURBO)
    heatmap[np.where(cam < 0.2)] = 0
    overlay_image = heatmap * 0.5 + image

    # Draw prediction boxes
    grayscale_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayscale_heatmap, int(0.8 * 255), 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 50 or rect[3] < 50:
            continue
        cv2.contourArea(c)
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, 'Prediction box', (x + 10, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                    lineType=cv2.LINE_AA)

    image = cv2.putText(image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)
    image = cv2.putText(image, "Press Q to quit !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)

    if annotations is not None and label_mapping is not None:
        image = draw_boxes_on_image(image, annotations, label_mapping)

    output_image = np.hstack([image, overlay_image])
    cv2.imwrite("output_image.png", output_image)
    cv2.imshow("Image", output_image)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        return


def main():
    args = parse_args()

    # Load model
    model = TomatoClassifier(args["model_dir"])
    prediction = model.has_tomatoes(args["image_path"])
    cam = model.get_cam(args["image_path"])

    annotations = None
    label_mapping = None
    if args["image_annotation_file"] is not None and args["label_mapping"] is not None:
        # Get image annotations
        logger.info("Loading image annotations ...")
        with open(args["image_annotation_file"], "r") as fp:
            image_annotations = json.load(fp)
        image_name = args["image_path"].split("/")[-1]
        annotations = image_annotations[image_name]
        # Get label mapping
        logger.info("Loading dataset label mapping ...")
        with open(args["label_mapping"], "r") as fp:
            label_mapping = pd.read_csv(fp)
        label_mapping = label_mapping.set_index("labelling_id").to_dict("index")

    # Visualize results
    visualize_prediction(args["image_path"], prediction, cam, annotations, label_mapping)


if __name__ == "__main__":
    main()
