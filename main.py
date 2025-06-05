'''
Detect known visual markers
Auto-rotate
Apply perspective correction
Crop with margin
Convert to b w
Export to landscape A5 PDF
'''

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import landscape, A5
from reportlab.pdfgen import canvas
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# === CONFIG ===
INPUT_FOLDER = "C:\Pictures"
OUTPUT_FOLDER = "C:\Pictures\Sorted"
MARGIN = 20
THREADS = 8
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
PARAMS = cv2.aruco.DetectorParameters()

def detect_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=PARAMS)
    if ids is not None and len(ids) >= 4:
        return dict(zip(ids.flatten(), corners))
    return None

def order_marker_corners(marker_dict):
    try:
        pts = np.array([
            marker_dict[0][0][0],
            marker_dict[1][0][0],
            marker_dict[2][0][0],
            marker_dict[3][0][0]
        ], dtype='float32')
        return pts
    except KeyError:
        return None

def four_point_transform(image, pts):
    width, height = 420, 595  # A5 landscape
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (width, height))

def crop_with_margin(img, margin=MARGIN):
    h, w = img.shape[:2]
    return img[margin:h-margin, margin:w-margin]

def convert_to_bw(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw

def save_as_pdf(bw_image, pdf_path):
    img_pil = Image.fromarray(bw_image).convert("RGB")
    temp_path = pdf_path.replace(".pdf", "_temp.jpg")
    img_pil.save(temp_path, "JPEG")

    c = canvas.Canvas(pdf_path, pagesize=landscape(A5))
    c.drawImage(temp_path, 0, 0, width=landscape(A5)[0], height=landscape(A5)[1])
    c.save()
    os.remove(temp_path)

def scan_document(image_path, output_path):
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠️ Cannot read: {image_path.name}")
            return

        marker_dict = detect_markers(image)
        if not marker_dict:
            print(f"❌ Markers not found: {image_path.name}")
            return

        pts = order_marker_corners(marker_dict)
        if pts is None:
            print(f"❌ Not all required markers (0–3): {image_path.name}")
            return

        warped = four_point_transform(image, pts)
        cropped = crop_with_margin(warped)
        bw = convert_to_bw(cropped)
        save_as_pdf(bw, output_path)
        print(f"✅ Done: {output_path.name}")

    except Exception as e:
        print(f"❌ Error processing {image_path.name}: {e}")

def process_folder_concurrently(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    image_extensions = (".jpg", ".jpeg", ".png")
    image_paths = [img for img in input_dir.glob("*") if img.suffix.lower() in image_extensions]

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = []
        for img_path in image_paths:
            output_path = output_dir / (img_path.stem + ".pdf")
            futures.append(executor.submit(scan_document, img_path, str(output_path)))

        # Optional: wait and gather errors
        for f in futures:
            f.result()

# === MAIN ===
if __name__ == "__main__":
    process_folder_concurrently(INPUT_FOLDER, OUTPUT_FOLDER)
