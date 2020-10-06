import argparse
import coloredlogs
import logging
import os
import json
import pandas as pd
import random

from shutil import copyfile
from src.util import visualize_images

coloredlogs.install(
    level='INFO',
    fmt="%(asctime)s %(name)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger("DatasetBuilder")


def parse_args():
    parser = argparse.ArgumentParser(description="Creating tfrecord dataset")
    parser.add_argument("--image_dir", required=True,
                        help="Path to directory containing images")
    parser.add_argument("--label_mapping", required=True,
                        help="Path to label mapping file")
    parser.add_argument("--image_annotation_file", required=True,
                        help="Path to file containing image annotations")
    parser.add_argument("--output_dir", required=True,
                        help="Path to folder containing converted labels")
    parser.add_argument("--display_images",
                        action="store_true", help="Display some images")

    return vars(parser.parse_args())


def generate_class_annotations(class_name, image_annotations, label_mapping):
    """Generating classification label for the class class_name from the original annotations
    """
    class_name = class_name.lower()
    class_image_annotations = {}
    image_paths = []
    binary_labels = []

    for image_path, annotations in image_annotations.items():
        class_present = False
        class_annotations = []
        for annotation in annotations:
            box_label = label_mapping[annotation["id"]]["labelling_name_en"]
            if class_name in box_label.lower():
                class_present = True
                class_annotations.append(annotation)

        if class_present:
            binary_labels.append(1)
        else:
            binary_labels.append(0)
        class_image_annotations[image_path] = class_annotations
        image_paths.append(image_path)

    return class_image_annotations, image_paths, binary_labels


def main():
    args = parse_args()

    # Get image annotations
    logger.info("Loading image annotations ...")
    with open(args["image_annotation_file"], "r") as fp:
        image_annotations = json.load(fp)

    # Randomly shuffling
    image_annotations = list(image_annotations.items())
    random.shuffle(image_annotations)
    image_annotations = dict(image_annotations)

    # Get label mapping
    logger.info("Loading dataset label mapping ...")
    with open(args["label_mapping"], "r") as fp:
        label_mapping = pd.read_csv(fp)
    label_mapping = label_mapping.set_index("labelling_id").to_dict("index")

    # Generate tomato annotations
    logger.info("Creating tomato annotations ...")
    tomato_annotations, image_paths, binary_labels = generate_class_annotations(
        "tomato", image_annotations, label_mapping)

    logger.info("Exporting tomato annotations ...")
    # Write converted annotations to file
    os.makedirs(args["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(args["output_dir"], "0"), exist_ok=True)
    os.makedirs(os.path.join(args["output_dir"], "1"), exist_ok=True)

    with open(os.path.join(args["output_dir"], "tomato_annotations.json"), "w") as fp:
        json.dump(tomato_annotations, fp, indent=4)

    for image_path, label in zip(image_paths, binary_labels):
        copyfile(os.path.join(args["image_dir"], image_path), os.path.join(
            args["output_dir"], "{}/{}".format(label, image_path)))

    logger.info("Build tomato dataset successfully !!!")
    if args["display_images"]:
        visualize_images(args["image_dir"], tomato_annotations, label_mapping)


if __name__ == "__main__":
    main()
