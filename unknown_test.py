import os
import cv2
import numpy as np

from insightface.app import FaceAnalysis


# ============================================================
# CONFIGURATION
# ============================================================

KHUSHBOO_DIR = r"E:\facieal_recoginition_projects\02_eigenfaces\dataset\Khushboo"

ABDULLAH_DIR = r"E:\facieal_recoginition_projects\01_haar_casacade\dataset\Abdullah Gul"

ADRIEN_DIR = r"E:\facieal_recoginition_projects\01_haar_casacade\dataset\Adrien Brody"

THRESHOLD = 0.70

MAX_KHUSHBOO_IMAGES = 20


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print("=" * 70)
print("LOADING INSIGHTFACE / ARCFACE")
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

print("ArcFace recognition model loaded successfully.")
print(f"Threshold: {THRESHOLD}")


# ============================================================
# FUNCTION: GET EMBEDDING
# ============================================================

def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"ERROR: Could not read {image_path}")
        return None

    # Current dataset contains cropped face images.
    # Resize directly to ArcFace input size.
    face_img = cv2.resize(
        image,
        (112, 112)
    )

    embedding = recognition_model.get_feat(face_img)

    embedding = embedding.flatten()

    # L2 normalization
    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None

    embedding = embedding / norm

    return embedding


# ============================================================
# BUILD KHUSHBOO DATABASE
# ============================================================

print("\n" + "=" * 70)
print("BUILDING KHUSHBOO FACE DATABASE")
print("=" * 70)

database_embeddings = []

image_files = [
    f for f in os.listdir(KHUSHBOO_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

image_files.sort()

image_files = image_files[:MAX_KHUSHBOO_IMAGES]

for filename in image_files:

    path = os.path.join(KHUSHBOO_DIR, filename)

    embedding = get_embedding(path)

    if embedding is not None:

        database_embeddings.append(embedding)

        print(f"{filename}: added to database")


database_embeddings = np.array(database_embeddings)

print("-" * 70)
print(
    f"Khushboo database: "
    f"{len(database_embeddings)} embeddings"
)


# ============================================================
# CREATE REFERENCE EMBEDDING
# ============================================================

# Average all known Khushboo embeddings.
# This creates a single representative face embedding.

reference_embedding = np.mean(
    database_embeddings,
    axis=0
)

# Normalize again after averaging.
reference_embedding = (
    reference_embedding /
    np.linalg.norm(reference_embedding)
)

print("Reference embedding created.")


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
# TEST FUNCTION
# ============================================================

def test_person(folder, person_name, max_images=10):

    print("\n" + "=" * 70)
    print(f"TESTING: {person_name}")
    print("=" * 70)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    files.sort()

    files = files[:max_images]

    matches = 0
    unknowns = 0

    scores = []

    for filename in files:

        path = os.path.join(folder, filename)

        embedding = get_embedding(path)

        if embedding is None:
            print(f"{filename}: ERROR")
            continue

        score = cosine_similarity(
            reference_embedding,
            embedding
        )

        scores.append(score)

        if score >= THRESHOLD:

            result = "MATCH"
            matches += 1

        else:

            result = "UNKNOWN"
            unknowns += 1

        print(
            f"{filename:<15} "
            f"Similarity: {score:.4f}   "
            f"Result: {result}"
        )

    print("-" * 70)

    if scores:

        print(f"Images tested : {len(scores)}")
        print(f"MATCH         : {matches}")
        print(f"UNKNOWN       : {unknowns}")
        print(f"Minimum score : {min(scores):.4f}")
        print(f"Maximum score : {max(scores):.4f}")
        print(f"Average score : {np.mean(scores):.4f}")


# ============================================================
# RUN TESTS
# ============================================================

# Genuine person
test_person(
    KHUSHBOO_DIR,
    "KHUSHBOO",
    max_images=20
)


# Unknown person 1
test_person(
    ABDULLAH_DIR,
    "ABDULLAH GUL",
    max_images=10
)


# Unknown person 2
test_person(
    ADRIEN_DIR,
    "ADRIEN BRODY",
    max_images=10
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("UNKNOWN PERSON TEST COMPLETED")
print("=" * 70)

print(f"Threshold used: {THRESHOLD}")