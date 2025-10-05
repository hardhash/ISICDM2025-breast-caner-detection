from Config import Config
import os
import cv2
import torch
from ultralytics import YOLO
from timm import create_model
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import csv
import json

def get_yolo(YOLO_WEIGHT_DIR=Config.YOLO_WEIGHT_DIR, IMG_SIZE=Config.IMG_SIZE, DEVICE=Config.DEVICE):
    print(f"Loading YOLO model (size {IMG_SIZE})...")
    weight_path = os.path.join(YOLO_WEIGHT_DIR, f"{IMG_SIZE}yolov8.pt")
    yolo_model = YOLO(weight_path)
    return yolo_model.to(DEVICE)

def get_cls(CLS_WEIGHT_DIR=Config.CLS_WEIGHT_DIR, IMG_SIZE=Config.IMG_SIZE, DEVICE=Config.DEVICE, NUM_CLASSES=Config.NUM_CLASSES, model_name=Config.cls_model):
    print(f"Loading EfficientNet classifiers (size {Config.IMG_SIZE})...")
    cls_models = []
    for fold in range(1, 6):
        WEIGHT_PATH = os.path.join(CLS_WEIGHT_DIR, f"{IMG_SIZE}_efficientnet_b0_fold_{fold}.pth")
        model = create_model(model_name, pretrained=False, num_classes=NUM_CLASSES, in_chans=1)
        checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        model.eval()
        cls_models.append(model)
    return cls_models

def get_transform(img_size=Config.IMG_SIZE):
    return transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # 输出 [1, H, W]
        transforms.Normalize(mean=[0.98755298], std=[0.026713483])
    ])

def infer(TEST_IMG_DIR=Config.TEST_IMG_DIR, NUM_CLASSES=Config.NUM_CLASSES,IMG_SIZE=Config.IMG_SIZE, DEVICE=Config.DEVICE):
    results = []
    test_images = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.png')])
    yolo_model = get_yolo()
    cls_models  = get_cls()
    for img_name in tqdm(test_images):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        orig_img = cv2.imread(img_path)
        h, w = orig_img.shape[:2]

        # Step 1: YOLO 推理
        results_yolo = yolo_model(
            orig_img,
            imgsz=Config.IMG_SIZE,
            conf=Config.YOLO_CONF_THRESH,
            verbose=False
        )
        pred = results_yolo[0].boxes

        if pred is None or len(pred) == 0:
            continue

        boxes_xyxy = pred.xyxy.cpu()  # [N, 4]
        confs = pred.conf.cpu()  # [N]

        # Step 2: 对每个框进行分类预测
        for box, det_conf in zip(boxes_xyxy, confs):
            x1, y1, x2, y2 = box.int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = orig_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # [H, W]

            total_logits = torch.zeros(NUM_CLASSES, device=DEVICE)
            transform = get_transform(IMG_SIZE)
            input_tensor = transform(crop_gray).unsqueeze(0).to(DEVICE)  # [1, 1, H, W]
            fold_logits = []

            for cls_model in cls_models:  # 遍历所有模型
                with torch.no_grad():
                    logits = cls_model(input_tensor)
                    fold_logits.append(logits)

            avg_logits = torch.mean(torch.cat(fold_logits, dim=0), dim=0)
            total_logits += avg_logits

            probs = F.softmax(total_logits, dim=0)
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()

            results.append({
                'image_name': img_name,
                'xmin': x1,
                'ymin': y1,
                'xmax': x2,
                'ymax': y2,
                'predicted_class': pred_class,
                'confidence': confidence
            })
    df = pd.DataFrame(results)
    return df.to_csv('infer-result.csv', index=False)

def postprocess(input_path=Config.infer_result_path, output_path=Config.postprocess_output_path):

    df = pd.read_csv(input_path)
    df['center_x'] = (df['xmin'] + df['xmax']) / 2
    df['center_y'] = (df['ymin'] + df['ymax']) / 2

    final_rows = []

    grouped = df.groupby(['image_name', 'predicted_class'])

    for (img_name, cls), group in grouped:
        coords = group[['center_x', 'center_y']].values
        confs = group['confidence'].values
        bboxes = group[['xmin', 'ymin', 'xmax', 'ymax']].values
        n_samples = len(group)

        if n_samples == 1:
            row = group.iloc[0].copy()
            final_rows.append(row.drop(['center_x', 'center_y']).to_dict())
            continue

        best_k = 1

        # 只有当样本数 >= 3 时，才尝试用 silhouette_score 选最优 k（k >=2）
        if n_samples >= 3:
            best_score = -1
            max_k = min(5, n_samples)  # 最多聚类数不超过样本数
            for k in range(2, max_k + 1):
                if k >= n_samples:
                    continue
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(coords)
                # 必须有至少两个簇（实际可能因数据全相同而只有1个簇）
                n_labels = len(np.unique(labels))
                if n_labels < 2 or n_labels >= n_samples:
                    continue
                try:
                    score = silhouette_score(coords, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                except ValueError:
                    continue

        if best_k == 1:
            labels = np.zeros(n_samples, dtype=int)
        else:
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(coords)

        for cluster_id in range(best_k):
            cluster_mask = (labels == cluster_id)
            cluster_confs = confs[cluster_mask]
            cluster_original_rows = group.iloc[cluster_mask].copy()
            max_idx = np.argmax(cluster_confs)
            best_row = cluster_original_rows.iloc[max_idx].copy()
            max_conf = cluster_confs[max_idx]
            best_row['confidence'] = max_conf
            final_rows.append(best_row.drop(['center_x', 'center_y']).to_dict())

    final_df = pd.DataFrame(final_rows)

    original_columns = [col for col in df.columns if col not in ['center_x', 'center_y']]
    final_df = final_df[original_columns]
    return final_df.to_csv(output_path, index=False)

def coco_json(csv_path = Config.postprocess_output_path, output_json_path = Config.output_json_path):
    results = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            image_id = row['image_name'].strip()
            category_id = int(row['predicted_class'])
            xmin = float(row['xmin'])
            ymin = float(row['ymin'])
            xmax = float(row['xmax'])
            ymax = float(row['ymax'])
            score = float(row['confidence'])
            width = xmax - xmin
            height = ymax - ymin
            results.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, width, height],
                "score": score
            })

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)



if __name__ == '__main__':
    results = infer()
    postprocess(input_path=Config.infer_result_path, output_path=Config.postprocess_output_path)
    coco_json(csv_path=Config.postprocess_output_path, output_json_path=Config.output_json_path)