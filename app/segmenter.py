import cv2
import numpy as np

# Ini digunakan untuk crop image (misal dalam 1 kata ada 4 huruf)
def tight_crop(img):
    coords = cv2.findNonZero(img)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

# Ini dgunakan untuk menambahkan margin pada image agar berbentuk persegi
def pad_to_square(img, margin=10):
    h, w = img.shape
    size = max(h, w) + margin * 2

    canvas = np.ones((size, size), dtype=np.uint8) * 255

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img
    return canvas

# Ini digunakan untuk segmentasi huruf dari canvas yang berisi beberapa huruf
def segment_letters(canvas):
    orig_h, orig_w = canvas.shape[:2]
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    initial_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 500: continue
        initial_boxes.append([x, y, w, h])

    if not initial_boxes: return []

    initial_boxes.sort(key=lambda b: b[0])

    merged_boxes = []
    if initial_boxes:
        curr = initial_boxes[0]
        for next_box in initial_boxes[1:]:
            x_overlap = max(0, min(curr[0] + curr[2], next_box[0] + next_box[2]) - max(curr[0], next_box[0]))
            
            if x_overlap > min(curr[2], next_box[2]) * 0.5:
                new_x = min(curr[0], next_box[0])
                new_y = min(curr[1], next_box[1])
                new_w = max(curr[0] + curr[2], next_box[0] + next_box[2]) - new_x
                new_h = max(curr[1] + curr[3], next_box[1] + next_box[3]) - new_y
                curr = [new_x, new_y, new_w, new_h]
            else:
                merged_boxes.append(curr)
                curr = next_box
        merged_boxes.append(curr)

    letters = []
    for box in merged_boxes:
        x, y, w, h = box
        crop = thresh[y:y+h, x:x+w]
        
        full_canvas_letter = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        center_y = orig_h // 2
        center_x = orig_w // 2
        
        start_y = center_y - (h // 2)
        start_x = center_x - (w // 2)
        
        end_y = start_y + h
        end_x = start_x + w
        full_canvas_letter[start_y:end_y, start_x:end_x] = crop

        full_canvas_letter = 255 - full_canvas_letter 
        
        letters.append((x, full_canvas_letter))

    return [l[1] for l in letters]