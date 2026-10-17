
import cv2
import time

from inference_sdk import InferenceHTTPClient

# ==========================================================
# SETTINGS
# ==========================================================

CONFIDENCE = 0.80

IMAGE_PATH = "zazu.png"
OUTPUT_PATH = "tank_detection_result.png"

API_KEY = "YOUR_ROBOFLOW_API_KEY"

TANK_MODEL_ID = "ayodeji-oyelakin/rail-tanker-detection-instant-1"

# ==========================================================
# CONNECT TO ROBOFLOW
# ==========================================================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

# ==========================================================
# LOAD IMAGE
# ==========================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

output = image.copy()

# ==========================================================
# TANK DETECTION
# ==========================================================

start = time.perf_counter()

results = client.infer(
    image,
    model_id=TANK_MODEL_ID
)

print("\n==============================")
print("TANK WAGON DETECTION")
print("==============================")

detections = 0

for tank in results.get("predictions", []):

    confidence = tank.get("confidence", 0)

    if confidence < CONFIDENCE:
        continue

    # Roboflow returns the box centre and dimensions.
    cx = tank["x"]
    cy = tank["y"]
    width = tank["width"]
    height = tank["height"]

    # Convert to bounding-box corner coordinates.
    x1 = max(0, int(cx - width / 2))
    y1 = max(0, int(cy - height / 2))
    x2 = min(image.shape[1], int(cx + width / 2))
    y2 = min(image.shape[0], int(cy + height / 2))

    if x2 <= x1 or y2 <= y1:
        continue

    detections += 1

    # Draw the tank bounding box.
    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Label the detection.
    cv2.putText(
        output,
        f"Tank {confidence:.2f}",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    print(f"Tank confidence: {confidence:.3f}")
    print(f"Bounding box: ({x1}, {y1}, {x2}, {y2})")

# ==========================================================
# FINISH
# ==========================================================

elapsed = time.perf_counter() - start

print("\n==============================")
print(f"Tanks detected: {detections}")
print(f"Processing time: {elapsed:.3f} seconds")
print("==============================")

if not cv2.imwrite(OUTPUT_PATH, output):
    raise RuntimeError("Could not save output image.")

print(f"Result saved to: {OUTPUT_PATH}")

cv2.imshow("Tank Wagon Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
