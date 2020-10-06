import cv2
import os


def draw_boxes_on_image(image, annotations, label_mapping):
    for annotation in annotations:
        if annotation["is_background"]:
            continue
        box_coordinates = annotation["box"]
        box_label = label_mapping[annotation["id"]]["labelling_name_en"]

        image = cv2.rectangle(image,
                              (box_coordinates[0], box_coordinates[1]),
                              (box_coordinates[0] + box_coordinates[2],
                               box_coordinates[1] + box_coordinates[3]),
                              (255, 0, 0), 2)

        cv2.putText(image, box_label, (box_coordinates[0] + 10, box_coordinates[1] + box_coordinates[3]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1,
                    lineType=cv2.LINE_AA)

    return image


def visualize_images(image_folder, image_annotations, label_mapping):
    for image_name, annotations in image_annotations.items():
        image = cv2.imread(os.path.join(image_folder, image_name))
        image = draw_boxes_on_image(image, annotations, label_mapping)
        cv2.putText(image, "Press anykey to continue, Q to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                    1, lineType=cv2.LINE_AA)
        cv2.imshow("Image", image)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
