import os
import numpy as np
import cv2

WIDTH, HEIGHT = 1920, 1080

CLASS_NAMES = ['pedestrian', 'cyclist', 'car', 'big_vehicle']

color_list = {'pedestrian': [0.000, 0.447, 0.741],
              'cyclist': [0.850, 0.325, 0.098],
              'car': [0.929, 0.694, 0.125],
              'big_vehicle': [0.494, 0.184, 0.556]
              }


img_dir = "datasets/Rope2D291/images/val"
pred_dir = "datasets/Rope2D291/preds_yolo_MONO_2D_int8/val"
vis_save_dir = "datasets/Rope2D291/vis_ti_quant_preds"
if not os.path.exists(vis_save_dir):
    os.makedirs(vis_save_dir)

imgs_file, preds_file = sorted(os.listdir(img_dir)), sorted(os.listdir(pred_dir))

for img_file, pred_file in zip(imgs_file, preds_file):
    img = cv2.imread(os.path.join(img_dir, img_file))
    pred_path = os.path.join(pred_dir, pred_file)

    if os.path.exists(pred_path):
        with open(pred_path, "r") as f:
            preds = [
                x.split() for x in f.read().strip().splitlines() if len(x)
            ]
            preds = np.array(preds, dtype=np.float32)


            for pred in preds:
                cls_id, score, cx, cy, w, h = pred
                score = round(float(score), 4)
                w, h = w * WIDTH, h * HEIGHT
                x1, y1, x2, y2 = int(cx * WIDTH - w / 2.), int(cy * HEIGHT - h / 2.), int(cx * WIDTH + w / 2.), int(cy * HEIGHT + h / 2.)
                obj_type = CLASS_NAMES[int(cls_id)]
                color_type = color_list[obj_type]
                color = (color_type[0] * 255, color_type[1] * 255, color_type[2] * 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # 标签的颜色
                txt_color = (0, 0, 0) if (sum(color_type) / len(color_type)) > 0.5 else (255, 255, 255)
                txt_bk_color = (color_type[0] * 255 * 0.7, color_type[1] * 255 * 0.7, color_type[2] * 255 * 0.7)
                text = f"{obj_type}:{score}"
                label_size, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(img, (x1, y1 - label_size[1] - 1), (x1 + label_size[0], y1), txt_bk_color, -1)
                cv2.putText(img, text, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, \
                            0.4, txt_color, thickness=1)

            cv2.imwrite(os.path.join(vis_save_dir, img_file), img)


