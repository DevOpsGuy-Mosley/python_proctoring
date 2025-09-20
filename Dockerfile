# Dockerfile pour le microservice Python Proctoring
FROM python:3.9-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Définition du répertoire de travail
WORKDIR /app

# Copie des requirements et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Création d'un utilisateur non-root
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Exposition du port
EXPOSE 8001

# Commande de démarrage
CMD ["python", "proctoring_service.py"]
