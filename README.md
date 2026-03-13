# Multi agent IA 
Ce projet a pour but de tester plusieurs multi-agent pour remplir les tâches qui m'occupe tous les jours

## Avertissement
Ce projet à vocation de faire des tests avec des slm.
Ce projet a été développer sur WSL², avec python 3.12 et ollama

## Structure du repo
Le repo est structuré par système multi agent. Dans chaque dossier le **README** donne les instructions pour installer les modèles ollama utilisés et  charger un environnement python fonctionnel


## Type de système multi agent
# Tuto
Le dossier est un tuto qui se retrouve sur beaucoup d'autres sites internet.
Il permet de créer un système multi agent pour écrire un article sur un sujet donné en entrée.
- Un agent Researcher qui permet de trouver les données contenu dans le modèle
- Un agent Writer qui permet d'écrire à partir des données récupérés par l'agent Search
- Un agent Critic qui permet de contrôler le retour de l'agent Writer. Il peut boucler avec lui en lui proposant des corrections nécessaires

# AgentActu

Le dossier est un système multi agent qui recherche sur internet les actualités sur un sujet donnée par l'utilisateur et qui va aggréger les données pour les ressortir en résumé avec les sources associées

- Un agent Search qui va chercher sur internet les actualités, il se base sur l'API duckduckgo
- Un agent Summary qui va agréger les données récupéré précédemment
- Un agent Validation qui permet de données un note aux résumés fournit précédemment

# Agent fiction 

En cours