import os
import cv2
import numpy as np
import pandas as pd
from itertools import combinations

from insightface.app import FaceAnalysis


# ============================================================
# PATHS
# ============================================================

KHUSHBOO_DIR = r"E:\facieal_recoginition_projects\02_eigenfaces\dataset\Khushboo"

ABDULLAH_DIR = r"E:\facieal_recoginition_projects\01_haar_casacade\dataset\Abdullah Gul"

ADRIEN_DIR = r"E:\facieal_recoginition_projects\01_haar_casacade\dataset\Adrien Brody"

OUTPUT_CSV = r"E:\facieal_recoginition_projects\10_insightface\threshold_results.csv"


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print("=" * 70)
print("LOADING ARCFACE")
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

print("Recognition model loaded successfully.")
print("Model:", type(recognition_model).__name__)


# ============================================================
# FUNCTION: GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(folder, person_name, max_images=20):

    print("\n" + "=" * 70)
    print(f"PROCESSING: {person_name}")
    print(f"FOLDER: {folder}")
    print("=" * 70)

    embeddings = []
    filenames = []

    image_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_files.sort()

    image_files = image_files[:max_images]

    for filename in image_files:

        path = os.path.join(folder, filename)

        image = cv2.imread(path)

        if image is None:
            print(f"{filename}: could not read")
            continue

        # Resize cropped face to ArcFace input size
        face_img = cv2.resize(
            image,
            (112, 112)
        )

        try:

            embedding = recognition_model.get_feat(face_img)

            embedding = embedding.flatten()

            # L2 normalization
            norm = np.linalg.norm(embedding)

            if norm == 0:
                print(f"{filename}: invalid embedding")
                continue

            embedding = embedding / norm

            embeddings.append(embedding)
            filenames.append(filename)

            print(f"{filename}: embedding generated")

        except Exception as e:

            print(f"{filename}: ERROR -> {e}")

    print("-" * 70)
    print(
        f"{person_name}: "
        f"{len(embeddings)}/{len(image_files)} images processed"
    )

    return np.array(embeddings), filenames


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

khushboo_embeddings, khushboo_files = generate_embeddings(
    KHUSHBOO_DIR,
    "Khushboo",
    max_images=20
)

abdullah_embeddings, abdullah_files = generate_embeddings(
    ABDULLAH_DIR,
    "Abdullah Gul",
    max_images=20
)

adrien_embeddings, adrien_files = generate_embeddings(
    ADRIEN_DIR,
    "Adrien Brody",
    max_images=20
)


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
# GENUINE SCORES
# ============================================================

genuine_scores = []

for i, j in combinations(range(len(khushboo_embeddings)), 2):

    score = cosine_similarity(
        khushboo_embeddings[i],
        khushboo_embeddings[j]
    )

    genuine_scores.append(score)


# ============================================================
# IMPOSTOR SCORES
# ============================================================

impostor_scores = []

# Khushboo vs Abdullah
for k in range(len(khushboo_embeddings)):

    for a in range(len(abdullah_embeddings)):

        score = cosine_similarity(
            khushboo_embeddings[k],
            abdullah_embeddings[a]
        )

        impostor_scores.append(score)


# Khushboo vs Adrien
for k in range(len(khushboo_embeddings)):

    for a in range(len(adrien_embeddings)):

        score = cosine_similarity(
            khushboo_embeddings[k],
            adrien_embeddings[a]
        )

        impostor_scores.append(score)


genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)


# ============================================================
# PRINT SCORE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SCORE SUMMARY")
print("=" * 70)

print("\nGENUINE SCORES")
print("-" * 70)
print(f"Comparisons : {len(genuine_scores)}")
print(f"Minimum     : {genuine_scores.min():.4f}")
print(f"Maximum     : {genuine_scores.max():.4f}")
print(f"Average     : {genuine_scores.mean():.4f}")
print(f"Median      : {np.median(genuine_scores):.4f}")
print(f"Std         : {genuine_scores.std():.4f}")


print("\nIMPOSTOR SCORES")
print("-" * 70)
print(f"Comparisons : {len(impostor_scores)}")
print(f"Minimum     : {impostor_scores.min():.4f}")
print(f"Maximum     : {impostor_scores.max():.4f}")
print(f"Average     : {impostor_scores.mean():.4f}")
print(f"Median      : {np.median(impostor_scores):.4f}")
print(f"Std         : {impostor_scores.std():.4f}")


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

results = []

thresholds = np.arange(
    0.00,
    1.001,
    0.01
)

for threshold in thresholds:

    # Genuine:
    # score >= threshold means MATCH
    genuine_matches = np.sum(
        genuine_scores >= threshold
    )

    genuine_unknown = np.sum(
        genuine_scores < threshold
    )

    # Impostor:
    # score >= threshold means FALSE MATCH
    false_matches = np.sum(
        impostor_scores >= threshold
    )

    correct_unknown = np.sum(
        impostor_scores < threshold
    )

    total_genuine = len(genuine_scores)
    total_impostor = len(impostor_scores)

    # False Rejection Rate
    frr = (
        genuine_unknown /
        total_genuine
    )

    # False Acceptance Rate
    far = (
        false_matches /
        total_impostor
    )

    total_correct = (
        genuine_matches +
        correct_unknown
    )

    total_comparisons = (
        total_genuine +
        total_impostor
    )

    accuracy = (
        total_correct /
        total_comparisons
    )

    results.append({
        "Threshold": round(threshold, 2),
        "FAR": far,
        "FRR": frr,
        "Accuracy": accuracy,
        "False_Acceptances": int(false_matches),
        "False_Rejections": int(genuine_unknown),
        "True_Acceptances": int(genuine_matches),
        "True_Rejections": int(correct_unknown)
    })


results_df = pd.DataFrame(results)


# ============================================================
# BEST THRESHOLD
# ============================================================

# Find threshold with minimum FAR + FRR
results_df["Error"] = (
    results_df["FAR"] +
    results_df["FRR"]
)

best_row = results_df.loc[
    results_df["Error"].idxmin()
]


# ============================================================
# PRINT THRESHOLD RESULTS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD CALIBRATION")
print("=" * 70)

print(
    "\n"
    f"{'Threshold':<12}"
    f"{'FAR':<12}"
    f"{'FRR':<12}"
    f"{'Accuracy':<12}"
)

print("-" * 50)

for _, row in results_df.iterrows():

    # Print useful range
    if (
        row["Threshold"] >= 0.50
        and row["Threshold"] <= 0.80
    ):

        print(
            f"{row['Threshold']:<12.2f}"
            f"{row['FAR'] * 100:<12.2f}"
            f"{row['FRR'] * 100:<12.2f}"
            f"{row['Accuracy'] * 100:<12.2f}"
        )


# ============================================================
# BEST THRESHOLD DETAILS
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDED THRESHOLD")
print("=" * 70)

print(
    f"Threshold          : "
    f"{best_row['Threshold']:.2f}"
)

print(
    f"FAR                : "
    f"{best_row['FAR'] * 100:.2f}%"
)

print(
    f"FRR                : "
    f"{best_row['FRR'] * 100:.2f}%"
)

print(
    f"Accuracy           : "
    f"{best_row['Accuracy'] * 100:.2f}%"
)

print(
    f"False Acceptances  : "
    f"{int(best_row['False_Acceptances'])}"
)

print(
    f"False Rejections   : "
    f"{int(best_row['False_Rejections'])}"
)

print(
    f"True Acceptances   : "
    f"{int(best_row['True_Acceptances'])}"
)

print(
    f"True Rejections    : "
    f"{int(best_row['True_Rejections'])}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(OUTPUT_CSV)

print("\nThreshold calibration completed.")