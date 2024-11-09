from yolov5.utils.metrics import bbox_iou

def calculate_accuracy(predictions, ground_truths, iou_threshold=0.5):
    correct_predictions = 0
    total_predictions = 0
    
    for preds, gts in zip(predictions, ground_truths):
        for pred_box in preds:
            total_predictions += 1
            for gt_box in gts:
                iou = bbox_iou(pred_box, gt_box)  # Calculate IoU
                if iou > iou_threshold and pred_box.class == gt_box.class:
                    correct_predictions += 1
                    break  # Only count one match per predicted box
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

# predictions and ground_truths should be lists of bounding boxes with class labels
accuracy = calculate_accuracy(predictions, ground_truths)
print("Model Accuracy:", accuracy)
