FROM python:3.10-slim

# REQUIRED LABELS FOR LANDSEER
LABEL org.opencontainers.image.dataset="CIFAR-10"
LABEL org.opencontainers.image.defense_stage="post_training"
LABEL org.opencontainers.image.defense_type="outlier_removal"
LABEL org.opencontainers.image.framework="pytorch"

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir torch torchvision pandas numpy tqdm scikit-learn

# 1. Copy the individual files
COPY main.py defense.py utils.py /app/

# 2. Copy the models directory
COPY models /app/models/

# Default command
CMD ["python", "main.py", "--input-dir", "/data", "--output", "/output", "--defense-type", "EP"]