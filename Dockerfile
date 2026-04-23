# Étape 1 : Build (installation des dépendances)
FROM python:3.10-slim AS builder

WORKDIR /app

# Installer pip-tools et autres utilitaires si besoin
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copier uniquement requirements.txt
COPY requirements.txt .

# Installer les dépendances dans un dossier temporaire
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Étape 2 : Final (image minimale)
FROM python:3.10-slim

WORKDIR /app

# Copier les dépendances depuis le builder
COPY --from=builder /install /usr/local

# Copier ton projet (le .dockerignore filtre les dossiers inutiles)
COPY . .

# Exposer le port par défaut de Streamlit
EXPOSE 8501

# Lancer l'application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
