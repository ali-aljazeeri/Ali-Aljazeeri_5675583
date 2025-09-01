import cv2
import os
import json
import random

# paths
image_folder = "./combined_car_images/images"
output_folder = "./combined_car_images/labels"
output_json = os.path.join(output_folder, "annotations.json")
circle_radius = 20
os.makedirs(output_folder, exist_ok=True)

# open annotations if available
# so that we do not re-do the work
if os.path.exists(output_json):
    with open(output_json, "r") as f:
        annotations = json.load(f)
else:
    annotations = {}

current_circles = []
colors = {}


def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]


def draw_circles(image, circles):
    for (x, y, r, cid) in circles:
        cv2.circle(image, (x, y), r, colors.get(cid, (0, 255, 0)), -1)


def message_overlay(image, message):
    overlay = image.copy()
    height, width = image.shape[:2]

    # background
    rect_height = 80
    cv2.rectangle(overlay, (0, height // 2 - rect_height // 2), (width, height // 2 + rect_height // 2), (0, 0, 0), -1)

    # text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height // 2 + text_size[1] // 2

    cv2.putText(overlay, message, (text_x, text_y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    cv2.imshow("Annotator", overlay)
    cv2.waitKey(4000)


def mouse_callback(event, x, y, flags, param):
    global current_circles

    if event == cv2.EVENT_LBUTTONDOWN:
        circle_id = str(len(current_circles))
        if circle_id not in colors:
            colors[circle_id] = generate_random_color()
        current_circles.append((x, y, circle_radius, circle_id))

    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, (cx, cy, r, cid) in enumerate(current_circles):
            if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2:
                current_circles.pop(i)
                break


def annotate_images():
    global current_circles

    # if empty
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if not image_files:
        print("No images found.")
        return

    index = 0
    while 0 <= index < len(image_files):
        img_file = image_files[index]
        current_circles = annotations.get(img_file, [])

        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not load image: {img_file}")
            index += 1
            continue

        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", mouse_callback)

        while True:
            display_img = img.copy()
            draw_circles(display_img, current_circles)
            cv2.imshow("Annotator", display_img)
            key = cv2.waitKey(20)

            # leave
            if key == ord('q'):
                annotations[img_file] = current_circles
                with open(output_json, "w") as f:
                    json.dump(annotations, f, indent=2)
                cv2.destroyAllWindows()
                return

            # up arrow (previous)
            if key == ord('q'):  # Quit
                annotations[img_file] = current_circles
                with open(output_json, "w") as f:
                    json.dump(annotations, f, indent=2)
                cv2.destroyAllWindows()
                return

            # Arrow Up (prev)
            elif key == 0:
                if index == 0:
                    message_overlay(display_img, "This is the FIRST image.")
                else:
                    annotations[img_file] = current_circles
                    index -= 1
                    break
            # Arrow Down (next)
            elif key == 1:
                if index == len(image_files) - 1:
                    message_overlay(display_img, "This is the LAST image.")
                else:
                    annotations[img_file] = current_circles
                    index += 1
                    break

    with open(output_json, "w") as f:
        json.dump(annotations, f, indent=2)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    annotate_images()
