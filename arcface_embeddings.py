import cv2
import numpy as np
from insightface.model_zoo import ArcFaceONNX
from insightface.app import FaceAnalysis


# --------------------------------------------------
# 1. Initialize InsightFace
# --------------------------------------------------

print("Initializing InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("InsightFace initialized successfully")


# --------------------------------------------------
# 2. Find the ArcFace recognition model
# --------------------------------------------------

recognition_model = None

for model in app.models.values():
    if hasattr(model, "get_feat") and hasattr(model, "input_size"):
        recognition_model = model
        break

if recognition_model is None:
    raise RuntimeError("ArcFace recognition model was not found.")

print("Recognition model found")
print("Input size:", recognition_model.input_size)


# --------------------------------------------------
# 3. Load Eigenfaces image
# --------------------------------------------------

image_path = (
    r"E:\facieal_recoginition_projects"
    r"\02_eigenfaces\dataset\Khushboo\1.jpg"
)

img = cv2.imread(image_path)

if img is None:
    raise RuntimeError("Could not load image.")

print("Image loaded successfully")
print("Original shape:", img.shape)


# --------------------------------------------------
# 4. Resize to ArcFace input size
# --------------------------------------------------

input_size = recognition_model.input_size

face_img = cv2.resize(img, input_size)

print("Resized shape:", face_img.shape)


# --------------------------------------------------
# 5. Generate ArcFace embedding
# --------------------------------------------------

embedding = recognition_model.get_feat(face_img)

embedding = embedding.flatten()

print("\n========== ARCFace RESULT ==========")
print("Embedding shape:", embedding.shape)
print("Embedding length:", len(embedding))

print("\nFirst 10 values:")
print(embedding[:10])

print("\nEmbedding norm:")
print(np.linalg.norm(embedding))

print("====================================")