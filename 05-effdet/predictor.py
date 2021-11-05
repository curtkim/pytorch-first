from effdet.bench import _post_process, _batch_detection
from ensemble_boxes import ensemble_boxes_wbf
import numpy as np
from typing import List, Tuple


def run_wbf(predictions, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = [(prediction["boxes"] / image_size).tolist()]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]

        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        boxes = boxes * (image_size - 1)
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())

    return bboxes, confidences, class_labels


def _postprocess_single_prediction_detections(detections, prediction_confidence_threshold):
    """
    하나의 torch.tensor를 boxes(column0~4), scores(column4), classes(column5)로 분리한다.
    self.prediction_confidence_threshold 보다 큰 경우만 추출

    :param detections:
    :return:
    """

    boxes = detections.detach().cpu().numpy()[:, :4]
    scores = detections.detach().cpu().numpy()[:, 4]
    classes = detections.detach().cpu().numpy()[:, 5]
    indexes = np.where(scores > prediction_confidence_threshold)[0]
    boxes = boxes[indexes]

    return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}


def post_process_detections(detections, img_size, wbf_iou_threshold, prediction_confidence_threshold):
    predictions = []
    for i in range(detections.shape[0]):
        predictions.append(
            _postprocess_single_prediction_detections(detections[i], prediction_confidence_threshold)
        )

    predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_wbf(
        predictions, image_size=img_size, iou_thr=wbf_iou_threshold
    )

    return predicted_bboxes, predicted_class_confidences, predicted_class_labels


def __rescale_bboxes(predicted_bboxes, image_sizes, img_size):
    scaled_bboxes = []
    for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
        im_h, im_w = img_dims

        if len(bboxes) > 0:
            scaled_bboxes.append(
                (
                        np.array(bboxes)
                        * [
                            im_w / img_size,
                            im_h / img_size,
                            im_w / img_size,
                            im_h / img_size,
                        ]
                ).tolist()
            )
        else:
            scaled_bboxes.append(bboxes)

    return scaled_bboxes


def predict(images, image_sizes: List[Tuple[int, int]], model, anchor_boxes, conf):
    class_out, box_out = model(images)

    # if eval mode, output detections for evaluation
    class_out_pp, box_out_pp, indices, classes = _post_process(
        class_out, box_out,
        num_levels=conf.num_levels,
        num_classes=conf.num_classes,
        max_detection_points=conf.max_detection_points)
    detections = _batch_detection(
        images.shape[0], class_out_pp, box_out_pp, anchor_boxes, indices, classes,
        None, None, #target['img_scale'], target['img_size'],
        max_det_per_image=conf.max_det_per_image,
        soft_nms=conf.soft_nms)

    predicted_bboxes, predicted_class_confidences, predicted_class_labels = \
        post_process_detections(detections, 512, 0.44, 0.2)     # TODO
    scaled_bboxes = __rescale_bboxes(predicted_bboxes=predicted_bboxes, image_sizes=image_sizes, img_size=512)
    return scaled_bboxes, predicted_class_labels, predicted_class_confidences
