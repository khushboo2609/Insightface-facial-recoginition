import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. Initialize InsightFace
# ============================================================

print("Initializing InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("InsightFace initialized successfully")


# ============================================================
# 2. Find ArcFace recognition model
# ============================================================

recognition_model = None

for model in app.models.values():
    if hasattr(model, "get_feat") and hasattr(model, "input_size"):
        recognition_model = model
        break

if recognition_model is None:
    raise RuntimeError("ArcFace recognition model not found.")

print("Recognition model found")
print("Input size:", recognition_model.input_size)


# ============================================================
# 3. Function to generate embedding
# ============================================================

def get_embedding(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not load: {image_path}")
        return None

    # Resize to ArcFace input
    face_img = cv2.resize(
        img,
        recognition_model.input_size
    )

    # Generate embedding
    embedding = recognition_model.get_feat(face_img)

    embedding = embedding.flatten()

    # L2 normalization
    norm = np.linalg.norm(embedding)

    if norm != 0:
        embedding = embedding / norm

    return embedding


# ============================================================
# 4. Dataset path
# ============================================================

dataset_path = (
    r"E:\facieal_recoginition_projects"
    r"\02_eigenfaces\dataset\Khushboo"
)


# ============================================================
# 5. Generate embeddings
# ============================================================

embeddings = []
image_names = []

print("\nGenerating embeddings...")
print("-" * 60)

for i in range(1, 51):

    image_path = f"{dataset_path}\\{i}.jpg"

    embedding = get_embedding(image_path)

    if embedding is not None:

        embeddings.append(embedding)
        image_names.append(f"{i}.jpg")

        print(f"{i}.jpg: embedding generated")


print("-" * 60)

print(f"Total embeddings generated: {len(embeddings)}")


# ============================================================
# 6. Convert to NumPy array
# ============================================================

embeddings = np.array(embeddings)

print("Embedding matrix shape:", embeddings.shape)


# ============================================================
# 7. Calculate pairwise cosine similarity
# ============================================================

similarity_matrix = cosine_similarity(embeddings)


# ============================================================
# 8. Remove self-comparisons
# ============================================================

similarity_values = []

for i in range(len(similarity_matrix)):

    for j in range(i + 1, len(similarity_matrix)):

        similarity_values.append(
            similarity_matrix[i][j]
        )


similarity_values = np.array(similarity_values)


# ============================================================
# 9. Statistics
# ============================================================

print("\n========== SIMILARITY RESULTS ==========")

print(
    "Number of comparisons:",
    len(similarity_values)
)

print(
    "Minimum similarity:",
    round(float(np.min(similarity_values)), 4)
)

print(
    "Maximum similarity:",
    round(float(np.max(similarity_values)), 4)
)

print(
    "Average similarity:",
    round(float(np.mean(similarity_values)), 4)
)

print(
    "Median similarity:",
    round(float(np.median(similarity_values)), 4)
)

print(
    "Standard deviation:",
    round(float(np.std(similarity_values)), 4)
)


# ============================================================
# 10. Display first 10 comparisons
# ============================================================

print("\n========== SAMPLE COMPARISONS ==========")

count = 0

for i in range(len(image_names)):

    for j in range(i + 1, len(image_names)):

        print(
            f"{image_names[i]} vs "
            f"{image_names[j]} = "
            f"{similarity_matrix[i][j]:.4f}"
        )

        count += 1

        if count == 10:
            break

    if count == 10:
        break


print("\n========================================")
print("ArcFace similarity test completed.")
print("========================================")