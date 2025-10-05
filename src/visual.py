from Config import Config
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def visualize(multi=True, single_image_name=None, visual_gap=Config.visual_gap):
    csv_path = Config.postprocess_output_path
    image_dir = Config.TEST_IMG_DIR
    df = pd.read_csv(csv_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    text_thickness = 8
    rectangle_thickness = 8

    label_colors = {
        '0': (0, 255, 0),
        '1': (255, 165, 0),
        '2': (255, 0, 0),
        '3': (0, 0, 255),
        '4': (255, 0, 255),
        '5': (0, 255, 255),
        '6': (128, 0, 128),
    }


    if multi:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()
        for i in range(9):
            image_name = f"ISICDM2025_test_{i + visual_gap:03d}.png"  # change 41 to any START number to visualize [START: START+9] result
            preds = df[df['image_name'] == image_name]
            img_path = Path(image_dir) / image_name
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            for _, row in preds.iterrows():
                xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                label = str(row['predicted_class'])
                confidence = row['confidence']

                if label in label_colors:
                    text_color = label_colors[label]
                else:
                    text_color = (0, 255, 0)

                label_with_confidence = f"{label} - {confidence:.4f}"

                cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), text_color, rectangle_thickness)
                cv2.putText(
                    img_rgb,
                    label_with_confidence,
                    (xmin, ymin - 10),
                    font,
                    font_scale,
                    text_color,
                    text_thickness,
                    cv2.LINE_AA
                )

            ax = axes[i]
            ax.imshow(img_rgb)
            ax.set_title(image_name)
            ax.axis('off')


    else:
        plt.figure(figsize=(10, 8))
        image_name = single_image_name
        preds = df[df['image_name'] == image_name]
        img_path = Path(image_dir) / image_name
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        for _, row in preds.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = str(row['predicted_class'])
            confidence = row['confidence']

            if label in label_colors:
                text_color = label_colors[label]
            else:
                text_color = (0, 255, 0)

            label_with_confidence = f"{label} - {confidence:.4f}"

            cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), text_color, rectangle_thickness)
            cv2.putText(
                img_rgb,
                label_with_confidence,
                (xmin, ymin - 10),
                font,
                font_scale,
                text_color,
                text_thickness,
                cv2.LINE_AA
            )
        plt.imshow(img_rgb)
        plt.title(image_name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Result Visualization')
    parser.add_argument('--multi', '-m', action='store_true',help='Muti-display Mode')
    parser.add_argument('--image', '-i', type=str,default='ISICDM2025_test_001.png',help='Single-display Mode')
    args = parser.parse_args()

    visualize(multi=args.multi, single_image_name=args.image)
