import torch

TRAIN_DIR = "dataset_processed/train"
VAL_DIR   = "dataset_processed/validation"
TEST_DIR = 'dataset_processed/test'
TEACHER_WEIGHTS = "model\\resnet50_pain.pth"

# Hyperparameters
IMG_SIZE = 224 # image size
PATCH_SIZE = 16 # number of patches
EMBED_DIM = 384 # Each token (patch, class, distillation) is represented by a 384-dimensional vector.
DEPTH = 6 # transformer has 3 stacked transformer blocks (layers). Each block = one full round of self attention , FFN
NUM_HEADS = 6  # Self-attention is split into 3 parallel attention mechanisms (heads). Each head looks at the same tokens but focuses on different relationships.
NUM_CLASSES = 2
BATCH_SIZE = 1 # The model processes 16 images at the same time before updating its weights.
EPOCHS = 10
LR = 3e-4
TEMPERATURE = 2.0 # It softens the teacher’s and student’s probability distributions during distillation. It controls how much uncertainty information the student learns from the teacher.
ATTN_SMOOTH_LAMBDA = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STUDENT_WEIGHTS = 'model/spatial_after_transformer_student_pain.pth'