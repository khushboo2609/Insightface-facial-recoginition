import cv2
from insightface.app import FaceAnalysis

# --------------------------------------------------
# 1. Initialize InsightFace
# --------------------------------------------------

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace initialized successfully")


# --------------------------------------------------
# 2. Load the Dlib dataset image
# --------------------------------------------------

image_path = r"E:\facieal_recoginition_projects\08_dlib\dataset\Khushboo\1.jpg"

img = cv2.imread(image_path)

if img is None:
    print("ERROR: Could not load image")
    print(image_path)
    exit()

print("Image loaded successfully")


# --------------------------------------------------
# 3. Detect faces using SCRFD
# --------------------------------------------------

faces = app.get(img)

print(f"Number of faces detected: {len(faces)}")


# --------------------------------------------------
# 4. Display information about each detected face
# --------------------------------------------------

for i, face in enumerate(faces):

    bbox = face.bbox
    det_score = face.det_score

    print(f"\nFace {i + 1}")
    print(f"Bounding box: {bbox}")
    print(f"Detection confidence: {det_score:.4f}")

    if face.kps is not None:
        print("5 facial landmarks detected")


# --------------------------------------------------
# 5. Draw bounding boxes
# --------------------------------------------------

for face in faces:

    bbox = face.bbox.astype(int)

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = f"Face: {face.det_score:.2f}"

    cv2.putText(
        img,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


# --------------------------------------------------
# 6. Save the result
# --------------------------------------------------

output_path = "results/scrfd_detection.jpg"

cv2.imwrite(output_path, img)

print(f"\nResult saved to: {output_path}")


# --------------------------------------------------
# 7. Display result
# --------------------------------------------------

cv2.imshow("InsightFace - SCRFD Detection", img)

print("\nPress any key to close the window.")

cv2.waitKey(0)
cv2.destroyAllWindows()
