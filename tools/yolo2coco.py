import os
import os.path as osp
import argparse
from multiprocessing.pool import Pool
from tqdm import tqdm
import json

from PIL import ExifTags, Image, ImageOps
import numpy as np

# Parameters
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])
VID_FORMATS.extend([f.upper() for f in VID_FORMATS])
# Get orientation exif tag
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k
        break

# class_names = ['trafficcone', 'tricyclist', 'van', 'cyclist', 'unknowns_movable', 'car', 'pedestrian',
#   'unknown_unmovable', 'bus', 'truck', 'barrow', 'motorcyclist']

class_names = ['pedestrian', 'cyclist', 'car', 'big_vehicle']

def check_image(im_file):
    '''Verify an image.'''
    nc, msg = 0, ""
    try:
        im = Image.open(im_file)
        im.verify()  # PIL verify
        im = Image.open(im_file)  # need to reload the image after using verify()
        shape = im.size  # (width, height)
        try:
            im_exif = im._getexif()
            if im_exif and ORIENTATION in im_exif:
                rotation = im_exif[ORIENTATION]
                if rotation in (6, 8):
                    shape = (shape[1], shape[0])
        except:
            im_exif = None
        if im_exif and ORIENTATION in im_exif:
            rotation = im_exif[ORIENTATION]
            if rotation in (6, 8):
                shape = (shape[1], shape[0])

        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(
                        im_file, "JPEG", subsampling=0, quality=100
                    )
                    msg += f"WARNING: {im_file}: corrupt JPEG restored and saved"
        return im_file, shape, nc, msg
    except Exception as e:
        nc = 1
        msg = f"WARNING: {im_file}: ignoring corrupt image: {e}"
        return im_file, None, nc, msg

def check_label_2d_files(args):
    img_path, lb_path = args
    nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
    try:
        if os.path.exists(lb_path):
            nf = 1  # label found
            with open(lb_path, "r") as f:
                labels = [
                    x.split() for x in f.read().strip().splitlines() if len(x)
                ]
                labels = np.array(labels, dtype=np.float32)
            if len(labels):
                assert all(
                    len(l) == 5 for l in labels
                ), f"{lb_path}: wrong label format."
                assert (
                    labels >= 0
                ).all(), f"{lb_path}: Label values error: all values in label file must > 0"
                assert (
                    labels[:, 1:] <= 1
                ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"

                _, indices = np.unique(labels, axis=0, return_index=True)
                if len(indices) < len(labels):  # duplicate row check
                    labels = labels[indices]  # remove duplicates
                    msg += f"WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"
                labels = labels.tolist()
            else:
                ne = 1  # label empty
                labels = []
        else:
            nm = 1  # label missing
            labels = []

        return img_path, labels, nc, nm, nf, ne, msg
    except Exception as e:
        nc = 1
        msg = f"WARNING: {lb_path}: ignoring invalid labels: {e}"
        return img_path, None, nc, nm, nf, ne, msg

def check_label_25d_files(args):
    img_path, lb_path = args
    nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
    try:
        if osp.exists(lb_path):
            nf = 1  # label found
            with open(lb_path, "r") as f:
                labels = [
                    x.split() for x in f.read().strip().splitlines() if len(x)
                ]
                labels = [
                    [float(lb) for lb in lbs] for lbs in labels
                ]

                labels = [
                    lbs for lbs in labels if abs(lbs[5]) >= 1e-6 and abs(lbs[6]) >= 1e-6 and abs(lbs[7]) >= 1e-6
                ]

                labels = np.array(labels, dtype=np.float32)

            if len(labels):
                assert all(
                    len(l) == 21 for l in labels
                ), f"{lb_path}: wrong label format."
                assert (
                        labels[:, [0, 1, 2, 3, 4]] >= 0
                ).all(), f"{lb_path}: Label values error: all values in label file must > 0"
                assert (
                        labels[:, 1:5] <= 1
                ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"

                _, indices = np.unique(labels, axis=0, return_index=True)
                if len(indices) < len(labels):  # duplicate row check
                    labels = labels[indices]  # remove duplicates
                    msg += f"WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"
                labels = labels.tolist()
            else:
                ne = 1  # label empty
                labels = []
        else:
            nm = 1  # label missing
            labels = []

        return img_path, labels, nc, nm, nf, ne, msg
    except Exception as e:
        nc = 1
        msg = f"WARNING: {lb_path}: ignoring invalid labels: {e}"
        return img_path, None, nc, nm, nf, ne, msg


def generate_coco_format_labels_2d(img_info, class_names, save_path):
    # for evaluation with pycocotools
    dataset = {"categories": [], "annotations": [], "images": []}
    for i, class_name in enumerate(class_names):
        dataset["categories"].append(
            {"id": i, "name": class_name, "supercategory": ""}
        )

    ann_id = 0
    print(f"Convert to COCO format")
    for i, (img_path, info) in enumerate(tqdm(img_info.items())):
        labels = info["labels"] if info["labels"] else []
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        img_w, img_h = info["shape"]
        dataset["images"].append(
            {
                "file_name": os.path.basename(img_path),
                "id": i,
                "width": img_w,
                "height": img_h,
            }
        )
        if labels:
            for label in labels:
                c, x, y, w, h = label[:5]
                # convert x,y,w,h to x1,y1,x2,y2
                x1 = (x - w / 2) * img_w
                y1 = (y - h / 2) * img_h
                x2 = (x + w / 2) * img_w
                y2 = (y + h / 2) * img_h
                # cls_id starts from 0
                cls_id = int(c)
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                dataset["annotations"].append(
                    {
                        "area": h * w,
                        "bbox": [x1, y1, w, h],
                        "category_id": cls_id,
                        "id": ann_id,
                        "image_id": i,
                        "iscrowd": 0,
                        # mask
                        "segmentation": [],
                    }
                )
                ann_id += 1

    with open(save_path, "w") as f:
        json.dump(dataset, f)
        print(
            f"Convert to COCO format finished. Resutls saved in {save_path}"
        )

def generate_coco_format_labels_25d(img_info, class_names, save_path):
    # for evaluation with pycocotools
    dataset = {"categories": [], "annotations": [], "images": []}
    for i, class_name in enumerate(class_names):
        dataset["categories"].append(
            {"id": i, "name": class_name, "supercategory": ""}
        )

    ann_id = 0
    print(f"Convert to COCO format")
    for i, (img_path, info) in enumerate(tqdm(img_info.items())):
        labels = info["labels"] if info["labels"] else []
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        img_w, img_h = info["shape"]
        dataset["images"].append(
            {
                "file_name": os.path.basename(img_path),
                "id": i,
                "width": img_w,
                "height": img_h,
            }
        )
        if labels:
            for label in labels:
                c, x, y, w, h = label[:5]
                # convert x,y,w,h to x1,y1,x2,y2
                x1 = (x - w / 2) * img_w
                y1 = (y - h / 2) * img_h
                x2 = (x + w / 2) * img_w
                y2 = (y + h / 2) * img_h
                # cls_id starts from 0
                cls_id = int(c)
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                H, W, L, X, Y, Z, ry, cos1, sin1, cos2, sin2, conf1, conf2, truncated, occluded, alpha = label[5:]
                dataset["annotations"].append(
                    {
                        "area": h * w,
                        "bbox": [x1, y1, w, h],
                        "category_id": cls_id,
                        "id": ann_id,
                        "image_id": i,
                        "iscrowd": 0,
                        "height": H,
                        "length": L,
                        "width": W,
                        "x": X,
                        "y": Y,
                        "z": Z,
                        "rotation_y": ry,
                        "cos1": cos1,
                        "sin1": sin1,
                        "cos2": cos2,
                        "sin2": sin2,
                        "conf1": conf1,
                        "conf2": conf2,
                        "truncated": truncated,
                        "occluded": occluded,
                        "alpha": alpha,
                        # mask
                        "segmentation": [],
                    }
                )
                ann_id += 1

    with open(save_path, "w") as f:
        json.dump(dataset, f)
        print(
            f"Convert to COCO format finished. Resutls saved in {save_path}"
        )

def main(args):
    data_path = args.data_path
    label_type = args.label_type
    TASKS = args.task
    for TASK in TASKS:
        imgs_dir = os.path.join(data_path, f"images/{TASK}")
        img_files = sorted(os.listdir(imgs_dir))
        img_paths = [os.path.join(imgs_dir, file) for file in img_files]

        labels_dir = os.path.join(data_path, f"labels_yolo_MONO_{label_type}/{TASK}")
        label_files = sorted(os.listdir(labels_dir))
        label_paths = [os.path.join(labels_dir, file) for file in label_files]

        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number corrupt, messages
        NUM_THREADS = min(16, os.cpu_count())
        print(f"Processing labels with {NUM_THREADS} process(es): ")

        img_info = {}

        nc, msgs = 0, []  # number corrupt, messages
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(
                pool.imap(check_image, img_paths),
                total=len(img_paths),
            )
            for img_path, shape_per_img, nc_per_img, msg in pbar:
                if nc_per_img == 0:  # not corrupted
                    img_info[img_path] = {"shape": shape_per_img}
                nc += nc_per_img
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{nc} image(s) corrupted"
        pbar.close()
        if msgs:
            print("\n".join(msgs))

        if label_type == "2D":
            with Pool(NUM_THREADS) as pool:
                pbar = pool.imap(
                    check_label_2d_files, zip(img_paths, label_paths)
                )
                pbar = tqdm(pbar, total=len(label_paths))
                for (
                        img_path,
                        labels_per_file,
                        nc_per_file,
                        nm_per_file,
                        nf_per_file,
                        ne_per_file,
                        msg,
                ) in pbar:
                    if nc_per_file == 0:
                        img_info[img_path]["labels"] = labels_per_file
                    else:
                        img_info.pop(img_path)
                    nc += nc_per_file
                    nm += nm_per_file
                    nf += nf_per_file
                    ne += ne_per_file
                    if msg:
                        msgs.append(msg)
                    pbar.desc = f"{nf} label(s) found, {nm} label(s) missing, {ne} label(s) empty, {nc} invalid label files"
            pbar.close()
            if msgs:
                print("\n".join(msgs))

        elif label_type == "2.5D":
            with Pool(NUM_THREADS) as pool:
                pbar = pool.imap(
                    check_label_25d_files, zip(img_paths, label_paths)
                )
                pbar = tqdm(pbar, total=len(label_paths))
                for (
                        img_path,
                        labels_per_file,
                        nc_per_file,
                        nm_per_file,
                        nf_per_file,
                        ne_per_file,
                        msg,
                ) in pbar:
                    if nc_per_file == 0:
                        img_info[img_path]["labels"] = labels_per_file
                    else:
                        img_info.pop(img_path)
                    nc += nc_per_file
                    nm += nm_per_file
                    nf += nf_per_file
                    ne += ne_per_file
                    if msg:
                        msgs.append(msg)
                    pbar.desc = f"{nf} label(s) found, {nm} label(s) missing, {ne} label(s) empty, {nc} invalid label files"
            pbar.close()
            if msgs:
                print("\n".join(msgs))

        save_dir = os.path.join(data_path, f"annotations_{label_type}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(
            save_dir, "instances_" + f"{TASK}" + ".json"
        )

        if label_type == "2D":
            generate_coco_format_labels_2d(
                img_info, class_names, save_path
            )
        elif label_type == "2.5D":
            generate_coco_format_labels_25d(
                img_info, class_names, save_path
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generating Datasets")
    parser.add_argument("--data-path", type=str, default="datasets/Rope3D", help="")
    # parser.add_argument("--data-path", type=str, default="datasets/Rope2D", help="")
    parser.add_argument("--label-type", type=str, default="2D", help="")
    parser.add_argument("--task", type=str, default=["val"], help="")
    args = parser.parse_args()
    main(args)