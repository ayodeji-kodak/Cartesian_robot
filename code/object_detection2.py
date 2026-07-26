import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

# ==========================================================
# SETTINGS
# ==========================================================

CONFIDENCE = 0.80

font_scale = 0.8
thickness = 2

# ==========================================================
# LOAD MODELS
# ==========================================================

tank_model = YOLO("tank_model.pt")
hatch_model = YOLO("hatch_model.pt")
hole_model = YOLO("hole_keypoint_model.pt")

# ==========================================================
# LOAD IMAGE
# ==========================================================

image_path = "zazu.png"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(image_path)

output = image.copy()

# ==========================================================
# START TIMER
# ==========================================================

start = time.perf_counter()

# ==========================================================
# STAGE 1
# DETECT TANK WAGON
# ==========================================================

tank_results = tank_model.predict(
    image,
    conf=CONFIDENCE,
    verbose=False
)[0]

print("\n==============================")
print("Tank Detection")
print("==============================")

# ==========================================================
# LOOP THROUGH EVERY TANK FOUND
# ==========================================================

for tank in tank_results.boxes.data.tolist():

    tx1, ty1, tx2, ty2, tank_conf, tank_class = tank

    tx1 = int(tx1)
    ty1 = int(ty1)
    tx2 = int(tx2)
    ty2 = int(ty2)

    tank_crop = image[ty1:ty2, tx1:tx2]

    cv2.rectangle(
        output,
        (tx1, ty1),
        (tx2, ty2),
        (0,255,0),
        2
    )

    cv2.putText(
        output,
        f"Tank {tank_conf:.2f}",
        (tx1,ty1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0,255,0),
        thickness
    )

    print(f"Tank: {tank_conf:.3f}")

    # ======================================================
    # STAGE 2
    # DETECT HATCH INSIDE TANK
    # ======================================================

    hatch_results = hatch_model.predict(
        tank_crop,
        conf=CONFIDENCE,
        verbose=False
    )[0]

    print("   Hatch Detection")

    for hatch in hatch_results.boxes.data.tolist():

        hx1, hy1, hx2, hy2, hatch_conf, hatch_class = hatch

        hx1 = int(hx1)
        hy1 = int(hy1)
        hx2 = int(hx2)
        hy2 = int(hy2)

        # Convert to image coordinates

        global_hx1 = tx1 + hx1
        global_hy1 = ty1 + hy1

        global_hx2 = tx1 + hx2
        global_hy2 = ty1 + hy2

        cv2.rectangle(
            output,
            (global_hx1,global_hy1),
            (global_hx2,global_hy2),
            (255,0,0),
            2
        )

        cv2.putText(
            output,
            f"Hatch {hatch_conf:.2f}",
            (global_hx1,global_hy1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255,0,0),
            thickness
        )

        print(f"      Hatch: {hatch_conf:.3f}")

        hatch_crop = tank_crop[hy1:hy2,hx1:hx2]

        # ==================================================
        # STAGE 3
        # KEYPOINT DETECTION
        # ==================================================

        hole_results = hole_model.predict(
            hatch_crop,
            conf=CONFIDENCE,
            verbose=False
        )[0]

        print("         Hole Detection")

        if hole_results.keypoints is not None:

            points = hole_results.keypoints.xy.cpu().numpy()

            for obj in points:

                for point in obj:

                    x = int(point[0])
                    y = int(point[1])

                    # Convert back into image coordinates

                    image_x = tx1 + hx1 + x
                    image_y = ty1 + hy1 + y

                    print(
                        f"            Hole Centre: ({image_x},{image_y})"
                    )

                    cv2.circle(
                        output,
                        (image_x,image_y),
                        6,
                        (0,0,255),
                        -1
                    )

                    cv2.putText(
                        output,
                        "Hole",
                        (image_x+8,image_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,255),
                        2
                    )

# ==========================================================
# FINISH
# ==========================================================

elapsed = time.perf_counter()-start

print("\n==============================")
print(f"Processing Time: {elapsed:.3f} seconds")
print("==============================")

cv2.imshow("Detection Pipeline",output)

cv2.imwrite("pipeline_result.png",output)

cv2.waitKey(0)

cv2.destroyAllWindows()