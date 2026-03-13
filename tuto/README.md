# 1 Installation ollama
## Linux / macOS
curl -fsSL https://ollama.ai/install.sh | sh

## Windows 
télécharger depuis https://ollama.ai/download

## Vérifier l'installation
ollama --version

## Télécharger les modèles nécessaires
ollama pull llama3.2        # Modèle principal (8B)
ollama pull mistral         # Modèle alternatif
ollama pull nomic-embed-text # Pour les embeddings

## Lancer le serveur Ollama (s'il ne tourne pas déjà)
ollama serve

# 2 Installation environnement python
** Attention une version de python >= 3.12 est nécessaire**

Installation si nécessaire de uv
## Linux / MacOS
curl -LsSf https://astral.sh/uv/install.sh | sh

## Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex

## Synchronisation de l'environnement python
uv sync

## Accès à l'environnement python 
source .venv/bin/activate

## Lancement du système multiagent
python3 examples/advanced_example.py