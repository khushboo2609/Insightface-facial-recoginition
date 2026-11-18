import cv2
from pathlib import Path
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

print("InsightFace initialized successfully\n")


# --------------------------------------------------
# 2. Dlib dataset path
# --------------------------------------------------

dataset_path = Path(
    r"E:\facieal_recoginition_projects\08_dlib\dataset\Khushboo"
)

images = sorted(
    dataset_path.glob("*.jpg"),
    key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem
)

print(f"Images found: {len(images)}")
print("-" * 50)


# --------------------------------------------------
# 3. Test each image
# --------------------------------------------------

detected_count = 0

for image_path in images[:30]:

    img = cv2.imread(str(image_path))

    if img is None:
        print(f"{image_path.name}: Could not load")
        continue

    faces = app.get(img)

    if len(faces) > 0:

        detected_count += 1

        best_score = max(face.det_score for face in faces)

        print(
            f"{image_path.name}: "
            f"{len(faces)} face(s) detected | "
            f"confidence = {best_score:.4f}"
        )

    else:

        print(f"{image_path.name}: NO FACE")


# --------------------------------------------------
# 4. Summary
# --------------------------------------------------

print("\n" + "=" * 50)

print(
    f"Detected faces in "
    f"{detected_count}/{min(30, len(images))} images"
)

print("=" * 50)