import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis


# ============================================================
# CONFIGURATION
# ============================================================

KHUSHBOO_PATH = Path(
    r"E:\facieal_recoginition_projects"
    r"\02_eigenfaces\dataset\Khushboo"
)

ABDULLAH_PATH = Path(
    r"E:\facieal_recoginition_projects"
    r"\01_haar_casacade\dataset\Abdullah Gul"
)

ADRIEN_PATH = Path(
    r"E:\facieal_recoginition_projects"
    r"\01_haar_casacade\dataset\Adrien Brody"
)

MAX_IMAGES = 20


# ============================================================
# 1. INITIALIZE INSIGHTFACE
# ============================================================

print("Initializing InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace initialized successfully")


# ============================================================
# 2. GET ARCFACE RECOGNITION MODEL DIRECTLY
# ============================================================

recognition_model = app.models["recognition"]

print(
    "Recognition model:",
    type(recognition_model).__name__
)

print(
    "ArcFace input size:",
    recognition_model.input_size
)


# ============================================================
# 3. GENERATE ARCFACE EMBEDDING
# ============================================================

def get_embedding(image_path):

    img = cv2.imread(str(image_path))

    if img is None:

        print(
            f"Could not read: {image_path}"
        )

        return None

    try:

        # Resize cropped face to ArcFace input
        face_img = cv2.resize(
            img,
            recognition_model.input_size
        )

        # Generate ArcFace embedding
        embedding = recognition_model.get_feat(
            face_img
        )

        embedding = embedding.flatten()

        # L2 normalization
        norm = np.linalg.norm(embedding)

        if norm != 0:

            embedding = embedding / norm

        return embedding

    except Exception as e:

        print(
            f"Error processing "
            f"{image_path.name}: {e}"
        )

        return None


# ============================================================
# 4. LOAD PERSON DATASET
# ============================================================

def load_person_embeddings(
    person_name,
    folder
):

    print("\n" + "=" * 60)

    print(
        f"Processing: {person_name}"
    )

    print(
        f"Folder: {folder}"
    )

    print("=" * 60)

    files = sorted(
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.jpeg")) +
        list(folder.glob("*.png"))
    )

    files = files[:MAX_IMAGES]

    embeddings = []
    names = []

    for image_path in files:

        embedding = get_embedding(
            image_path
        )

        if embedding is not None:

            embeddings.append(
                embedding
            )

            names.append(
                image_path.name
            )

            print(
                f"{image_path.name}: "
                f"embedding generated"
            )

        else:

            print(
                f"{image_path.name}: "
                f"embedding failed"
            )

    print("-" * 60)

    print(
        f"{person_name}: "
        f"{len(embeddings)}/{len(files)} "
        f"images processed"
    )

    return np.array(embeddings), names


# ============================================================
# 5. COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:

        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ============================================================
# 6. LOAD DATASETS
# ============================================================

khushboo_embeddings, khushboo_names = \
    load_person_embeddings(
        "Khushboo",
        KHUSHBOO_PATH
    )


abdullah_embeddings, abdullah_names = \
    load_person_embeddings(
        "Abdullah Gul",
        ABDULLAH_PATH
    )


adrien_embeddings, adrien_names = \
    load_person_embeddings(
        "Adrien Brody",
        ADRIEN_PATH
    )


# ============================================================
# 7. GENUINE COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("GENUINE COMPARISON")
print("KHUSHBOO VS KHUSHBOO")
print("=" * 70)

genuine_scores = []

for i in range(
    len(khushboo_embeddings)
):

    for j in range(
        i + 1,
        len(khushboo_embeddings)
    ):

        score = cosine_similarity(
            khushboo_embeddings[i],
            khushboo_embeddings[j]
        )

        genuine_scores.append(score)


genuine_scores = np.array(
    genuine_scores
)


if len(genuine_scores) > 0:

    print(
        "Comparisons:",
        len(genuine_scores)
    )

    print(
        "Minimum:",
        f"{genuine_scores.min():.4f}"
    )

    print(
        "Maximum:",
        f"{genuine_scores.max():.4f}"
    )

    print(
        "Average:",
        f"{genuine_scores.mean():.4f}"
    )

    print(
        "Median:",
        f"{np.median(genuine_scores):.4f}"
    )

    print(
        "Std:",
        f"{genuine_scores.std():.4f}"
    )


# ============================================================
# 8. IMPOSTOR COMPARISON
# ============================================================

def compare_with_person(
    unknown_embeddings,
    unknown_name
):

    print("\n")
    print("=" * 70)

    print(
        f"IMPOSTOR COMPARISON"
    )

    print(
        f"KHUSHBOO VS {unknown_name.upper()}"
    )

    print("=" * 70)

    scores = []

    for khushboo_embedding in \
            khushboo_embeddings:

        for unknown_embedding in \
                unknown_embeddings:

            score = cosine_similarity(
                khushboo_embedding,
                unknown_embedding
            )

            scores.append(score)

    scores = np.array(scores)

    if len(scores) == 0:

        print("No comparisons available.")

        return scores

    print(
        "Comparisons:",
        len(scores)
    )

    print(
        "Minimum:",
        f"{scores.min():.4f}"
    )

    print(
        "Maximum:",
        f"{scores.max():.4f}"
    )

    print(
        "Average:",
        f"{scores.mean():.4f}"
    )

    print(
        "Median:",
        f"{np.median(scores):.4f}"
    )

    print(
        "Std:",
        f"{scores.std():.4f}"
    )

    return scores


# ============================================================
# 9. RUN IMPOSTOR TESTS
# ============================================================

abdullah_scores = compare_with_person(
    abdullah_embeddings,
    "Abdullah Gul"
)


adrien_scores = compare_with_person(
    adrien_embeddings,
    "Adrien Brody"
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL ARCFACE SUMMARY")
print("=" * 70)

print(
    "Khushboo images:",
    len(khushboo_embeddings)
)

print(
    "Abdullah Gul images:",
    len(abdullah_embeddings)
)

print(
    "Adrien Brody images:",
    len(adrien_embeddings)
)

if len(genuine_scores) > 0:

    print(
        "\nGenuine minimum:",
        f"{genuine_scores.min():.4f}"
    )

if len(abdullah_scores) > 0:

    print(
        "Abdullah maximum:",
        f"{abdullah_scores.max():.4f}"
    )

if len(adrien_scores) > 0:

    print(
        "Adrien maximum:",
        f"{adrien_scores.max():.4f}"
    )

print("=" * 70)

print(
    "\nArcFace genuine/impostor "
    "experiment completed."
)