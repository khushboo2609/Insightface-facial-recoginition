import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ============================================================
# CONFIGURATION
# ============================================================

KHUSHBOO_DIR = r"E:\facieal_recoginition_projects\02_eigenfaces\dataset\Khushboo"

THRESHOLD = 0.70

CAMERA_INDEX = 0


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print("=" * 70)
print("LOADING INSIGHTFACE")
print("=" * 70)

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

recognition_model = app.models["recognition"]

print("InsightFace loaded successfully.")
print("Detection model :", type(app.models["detection"]).__name__)
print("Recognition model:", type(recognition_model).__name__)
print("Threshold       :", THRESHOLD)


# ============================================================
# FUNCTION: GENERATE KHUSHBOO REFERENCE EMBEDDING
# ============================================================

def build_reference_embedding():

    print("\n" + "=" * 70)
    print("BUILDING KHUSHBOO REFERENCE")
    print("=" * 70)

    embeddings = []

    image_files = [
        f for f in __import__("os").listdir(KHUSHBOO_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_files.sort()

    for filename in image_files:

        path = __import__("os").path.join(
            KHUSHBOO_DIR,
            filename
        )

        image = cv2.imread(path)

        if image is None:
            print(f"{filename}: could not read")
            continue

        # Dataset contains cropped faces
        face_img = cv2.resize(
            image,
            (112, 112)
        )

        try:

            embedding = recognition_model.get_feat(
                face_img
            )

            embedding = embedding.flatten()

            norm = np.linalg.norm(embedding)

            if norm == 0:
                continue

            embedding = embedding / norm

            embeddings.append(embedding)

            print(f"{filename}: added")

        except Exception as e:

            print(f"{filename}: ERROR - {e}")

    if len(embeddings) == 0:

        raise RuntimeError(
            "No Khushboo embeddings could be generated."
        )

    embeddings = np.array(embeddings)

    # Average all embeddings
    reference = np.mean(
        embeddings,
        axis=0
    )

    # Normalize averaged embedding
    reference = reference / np.linalg.norm(reference)

    print("-" * 70)
    print(
        f"Reference created using "
        f"{len(embeddings)} images."
    )

    return reference


# ============================================================
# BUILD REFERENCE
# ============================================================

reference_embedding = build_reference_embedding()


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    return float(
        np.dot(a, b) /
        (
            np.linalg.norm(a) *
            np.linalg.norm(b)
        )
    )


# ============================================================
# START CAMERA
# ============================================================

print("\n" + "=" * 70)
print("STARTING WEBCAM")
print("=" * 70)

print("Press Q to quit.")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# ============================================================
# REAL-TIME RECOGNITION
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # Detect faces
    faces = app.get(frame)

    if len(faces) == 0:
        print("No face detected", end="\r")

    else:

        print(
            f"\nDetected {len(faces)} face(s)"
        )

        for face_index, face in enumerate(faces, start=1):

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            bbox = face.bbox.astype(int)

            x1, y1, x2, y2 = bbox

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # ------------------------------------------------
            # Get ArcFace embedding
            # ------------------------------------------------

            embedding = face.embedding

            if embedding is None:

                print(
                    f"Face {face_index}: "
                    "embedding unavailable"
                )

                continue

            embedding = embedding.astype(
                np.float32
            )

            # Normalize
            norm = np.linalg.norm(embedding)

            if norm == 0:

                print(
                    f"Face {face_index}: "
                    "invalid embedding"
                )

                continue

            embedding = embedding / norm

            # ------------------------------------------------
            # Calculate similarity
            # ------------------------------------------------

            similarity = cosine_similarity(
                reference_embedding,
                embedding
            )

            # ------------------------------------------------
            # Classification
            # ------------------------------------------------

            if similarity >= THRESHOLD:

                label = "KHUSHBOO"
                result_text = "MATCH"

            else:

                label = "UNKNOWN"
                result_text = "UNKNOWN"

            # ------------------------------------------------
            # Console output
            # ------------------------------------------------

            print(
                f"Face {face_index} | "
                f"Similarity: {similarity:.4f} | "
                f"Result: {result_text}"
            )

            # ------------------------------------------------
            # Text for webcam
            # ------------------------------------------------

            display_text = (
                f"{label} "
                f"{similarity:.2f}"
            )

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            cv2.putText(
                frame,
                display_text,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # Display webcam
    cv2.imshow(
        "InsightFace ArcFace Recognition",
        frame
    )

    # Q = quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("WEBCAM RECOGNITION STOPPED")
print("=" * 70)