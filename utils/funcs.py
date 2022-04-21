from utils.librarys import *

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def eval_model(model, data_loader):

    model.eval()
    map101 = []
    for images, targets in data_loader:
        
        images = [image.to(device) for image in images]
        targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

        # put the model in evaluation mode
        with torch.no_grad():
            predictions = model(images)
        
        for j in range(len(predictions)):
            idx_text = (targets[j]['labels'] == 1)
            tg_box = targets[j]["boxes"]
            tg_box = tg_box[idx_text]

            idx_text = (predictions[j]['labels'] == 1)
            pred_box = predictions[j]["boxes"]
            pred_box = pred_box[idx_text]

            ious_score = bbox_overlaps(pred_box.cpu(), tg_box).numpy()
            m = np.mean(ious_score[range(len(ious_score)), np.argmax(ious_score, -1)] >= 0.5)
            map101.append(m)

    return np.mean(map101)

def process_data(root='datasets/CROP/', mask_json='datasets/Skin_segmentation_final.json'):
    root = root

    with open(mask_json) as json_file:
        data = json.load(json_file)['_via_img_metadata']

    keys_data = list(data.keys()) # filename

    data_label = [] # filename, points
    for key in keys_data:
        if len(data[key]["regions"]) <= 0:
            continue
        filename = root + data[key]["filename"]
        points = [] # [[region1], [region2], ...]
        for r in data[key]["regions"]:
            point_x = r["shape_attributes"]["all_points_x"]
            point_y = r["shape_attributes"]['all_points_y']
            points.append(list(zip(point_x, point_y)))
        data_label.append((filename, points))

    split = round(len(data_label) * 0.8)
    train_label = data_label[:split]
    val_label = data_label[split:]
    return train_label, val_label