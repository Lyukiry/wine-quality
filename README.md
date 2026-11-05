# wine-quality

# 🍷 Classification de la qualité du vin

Projet **Machine Learning 1** (EFREI, 2026-2027).

**Objectif** : prédire si un vin est **bon** (note de qualité ≥ 7/10) à partir de ses caractéristiques physico-chimiques, en comparant plusieurs modèles de classification supervisée.

## Équipe

- _Lucie Le Saux_
- _Victor Lassalle_

## Données

Jeu de données **Wine Quality** (UCI, Cortez et al., 2009) : vins portugais *Vinho Verde*.

| Fichier | Contenu |
|---|---|
| `winequality-red.csv` | 1 599 vins rouges |
| `winequality-white.csv` | 4 898 vins blancs |

- Séparateur **`;`** (et non `,`).
- 11 variables physico-chimiques continues : acidité fixe, acidité volatile, acide citrique, sucre résiduel, chlorures, SO₂ libre, SO₂ total, densité, pH, sulfates, alcool.
- `quality` : note sensorielle de 0 à 10, la variable à prédire.
- Les fichiers ne précisent pas si un vin est rouge ou blanc. On ajoute donc une colonne `Type` (`red` / `white`) avant de les fusionner.

## Structure du dépôt

```
wine-quality/
├── data/                  # fichiers CSV bruts (jamais modifiés)
├── notebooks/             # notebooks d'analyse
├── reports/
│   └── figures/           # graphiques exportés pour le rapport
├── requirements.txt       # bibliothèques Python et leurs versions
└── README.md
```

## Installation

```bash
git clone <url-du-depot>
cd wine-quality
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Ensuite, dans VS Code, ouvrir un notebook et sélectionner le kernel `.venv` en haut à droite.

> Le dossier `.venv/` est ignoré par git : chaque membre crée le sien.

## Avancement

- [x] Mise en place du dépôt, de l'environnement virtuel et des dépendances
- [x] Chargement et fusion des deux fichiers (colonne `Type` ajoutée)
- [x] Statistiques descriptives
- [x] Valeurs manquantes et doublons
- [x] Détection des valeurs aberrantes (méthode IQR)
- [x] Décision sur les valeurs aberrantes
- [ ] Encodage de `Type` et création de la cible binaire (Victor)
- [ ] Analyse exploratoire : distributions, boxplots, corrélations (victor)
- [ ] Sélection de variables (victor)
- [ ] Modélisation (5 modèles) et évaluation
- [ ] Optimisation des hyperparamètres (Grid Search / Random Search)
- [ ] Rapport technique

## Résultats du prétraitement

### Statistiques descriptives

- **6 497 vins** au total, 12 colonnes numériques + `Type`.
- **Échelles très hétérogènes** : la densité varie de 0,987 à 1,039, le SO₂ total de 6 à 440. Il faudra standardiser les données pour les modèles qui reposent sur des distances (k-NN, SVM) ou sur une régularisation (régression logistique).
- Pour plusieurs variables, le **maximum est très éloigné du 3ᵉ quartile** (sucre résiduel, SO₂ libre, chlorures, sulfates), ce qui signale des valeurs extrêmes et des distributions asymétriques.

### Valeurs manquantes

Aucune valeur manquante.

### Doublons

**1 177 doublons exacts** (18 % des lignes) ont été **supprimés**. Il reste **5 320 vins**.

Raison : si une même observation se retrouve à la fois dans le jeu d'entraînement et dans le jeu de test, le modèle la « reconnaît » et le score de test est artificiellement gonflé (**fuite de données**). La suppression est faite **avant** le découpage train/test.

### Valeurs aberrantes (méthode IQR)

Une valeur est considérée comme aberrante si elle sort de l'intervalle [Q1 − 1,5 × IQR ; Q3 + 1,5 × IQR], avec IQR = Q3 − Q1.

| Variable | Outliers | % |
|---|---|---|
| fixed acidity | 304 | 5,71 |
| volatile acidity | 279 | 5,24 |
| chlorides | 237 | 4,45 |
| sulphates | 163 | 3,06 |
| citric acid | 143 | 2,69 |
| residual sugar | 141 | 2,65 |
| pH | 49 | 0,92 |
| free sulfur dioxide | 44 | 0,83 |
| total sulfur dioxide | 10 | 0,19 |
| density | 3 | 0,06 |
| alcohol | 1 | 0,02 |

- Au plus **5,7 % d'outliers par variable**, mais **1 094 lignes (20,6 %)** ont au moins une valeur aberrante sur l'une des 11 variables.
- **Aucune valeur physiquement impossible** : ce sont des vins réels qui sortent de l'ordinaire (vins moelleux, rouges très acides…), pas des erreurs de mesure.
- **Le statut d'outlier dépend du groupe de référence.** En calculant les bornes séparément pour les rouges et les blancs, le nombre d'outliers diminue pour l'acidité fixe (304 → 147) et l'acidité volatile (279 → 152), mais augmente pour d'autres variables (SO₂ total : 10 → 59, densité : 3 → 41). Le seuil IQR est une convention, pas une preuve d'erreur.

**Décision** : aucune supressions des valeurs aberantes car la perte de donner est trop important, 26% du data set.

## Travail en équipe

- Ne jamais travailler directement sur `main` : une branche par tâche.
  ```bash
  git switch main && git pull
  git switch -c nom-de-la-branche
  # ... travail ...
  git add <fichiers>
  git commit -m "Description courte"
  git push -u origin nom-de-la-branche
  ```
- Ouvrir une **Pull Request** sur GitHub ; un autre membre relit avant de fusionner.
- Les données brutes de `data/` ne sont jamais modifiées : toutes les transformations se font dans le notebook.