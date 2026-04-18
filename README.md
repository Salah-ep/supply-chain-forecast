# Supply Chain Demand Forecasting

Projet de prévision de la demande en Supply Chain avec Python et Machine Learning.
Réalisé dans le cadre d'une préparation au stage IA/Data chez Stellantis.

## Objectif
Analyser des données de ventes réelles et entraîner un modèle capable de
prévoir la demande future — un cas d'usage central en Supply Chain.

## Dataset
Télécharger train.csv depuis Kaggle :
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
Et le placer dans le dossier data/

## Installation
pip install -r requirements.txt

## Lancer le projet
python main.py

## Structure
supply_chain_forecast/
├── data/               → Données brutes (non trackées par Git)
├── notebooks/          → Exploration et analyse
├── src/                → Code source Python
│   ├── preprocessing.py   → Nettoyage et préparation des données
│   ├── eda.py             → Exploration et visualisation
│   └── model.py           → Entraînement et évaluation du modèle
├── outputs/            → Graphiques et résultats générés
├── main.py             → Point d'entrée du projet
├── requirements.txt    → Dépendances Python
└── .gitignore          → Fichiers ignorés par Git