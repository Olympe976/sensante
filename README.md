---
title: Sensante
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# Sensante

Assistant de pre-diagnostic medical pour le Senegal.

## Description

SenSante utilise le Machine Learning pour aider au pre-diagnostic des maladies courantes (paludisme, grippe, typhoide) a partir des symptomes du patient.

## Démo en ligne
https://amo-enigma-sensante.hf.space/


## Stack
- scikit-learn (modèle ML)
- FastAPI (API REST)
- Tailwind CSS (frontend responsive)
- Groq / Llama 3 (explication LLM)
- Docker (conteneurisation)

## Structure du projet

- `data/` : Donnees patients (CSV)
- `models/` : Modele ML serialise
- `api/` : API FastAPI
- `frontend/` : Inteface web
- `notebooks/` : Scripts d'exploration

## Auteur

Mahuna Olympe ATCHATIN - L2 GLSI - ESP/UCAD

## Cours

Integration de Modeles IA - Dr. El Hadji Bassirou TOURE