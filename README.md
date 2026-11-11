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
- [x] Encodage de `Type` et création de la cible binaire
- [x] Analyse exploratoire : distributions, boxplots, corrélations
- [x] Sélection de variables (victor)
- [x] Modélisation (5 modèles) et évaluation
- [x] Optimisation des hyperparamètres (Grid Search / Random Search)
- [ ] Rapport technique
- [ ] Bonus : tests unitaires, Docker, application web

## Résumé de la démarche
 
### 1. Prétraitement
 
- **6 497 vins**, aucune valeur manquante.
- **1 177 doublons exacts** (18 %) supprimés pour éviter une **fuite de données** entre train et test. Il reste **5 320 vins**.
- **Valeurs aberrantes** (méthode IQR) : au plus 5,7 % par variable, mais 20,6 % des vins en ont au moins une. **Nous les conservons** : aucune valeur n'est physiquement impossible (vins moelleux, rouges très acides…), et les supprimer ferait perdre un cinquième des données, dont une partie des bons vins.
- **Encodage** : `Type` → `Type_encoded` (0 = rouge, 1 = blanc) ; cible `quality_binary` = 1 si `quality` ≥ 7.
- **Déséquilibre** : 19 % de bons vins. Un modèle qui répondrait toujours « mauvais » aurait 81 % d'accuracy : on évalue donc surtout avec le **F1** et l'**AUC**, avec un découpage et une validation croisée **stratifiés**.
- **Découpage** : 80 % entraînement (4 256 vins), 20 % test (1 064 vins). Le test n'est utilisé qu'une seule fois, à la fin.
### 2. Analyse exploratoire
 
- Variables les plus liées à la qualité : **alcohol** (+0,47), **density** (−0,33), **volatile acidity** (−0,27), **chlorides** (−0,20).
- Paires redondantes : SO₂ libre / SO₂ total (+0,72), density / alcohol (−0,67), density / residual sugar (+0,52).
### 3. Sélection de variables
 
Trois familles de méthodes : **filtre** (corrélation, information mutuelle), **wrapper** (RFECV) et **embarquée** (Lasso, Ridge).
 
**10 variables retenues sur 12** : on retire `citric acid` et `Type_encoded`, les plus faibles selon la synthèse des méthodes. Leur retrait ne dégrade pas les meilleurs modèles. Un sous-ensemble plus réduit (3 variables, Lasso) convenait à la régression logistique mais dégradait fortement les modèles non linéaires.
 
### 4. Modélisation et évaluation (validation croisée 5-fold)
 
| Modèle | F1 | AUC | Diagnostic |
|---|---|---|---|
| SVM (RBF) | 0,543 | 0,841 | Bon compromis biais/variance |
| Régression logistique | 0,533 | 0,831 | Stable, léger biais (frontière linéaire) |
| Naive Bayes | 0,497 | 0,785 | Biais : hypothèse d'indépendance violée |
| k-NN | 0,436 | 0,783 | Variance, faible rappel |
| Arbre de décision | 0,421 | 0,642 | Sur-apprentissage massif (F1 train = 1,0) |
 
*Paramètres par défaut.*
 
### 5. Optimisation
 
- **Grid Search** et **Random Search** donnent des scores équivalents (écarts < 0,007).
- Gain principal pour l'**arbre** (F1 0,421 → 0,513) : limiter la taille des feuilles corrige son sur-apprentissage.
- Le Random Search est plus économe quand l'espace de recherche est grand (arbre : même score avec 20 combinaisons au lieu de 60).
### 6. Modèle final : régression logistique
 
Choisie car **statistiquement équivalente au SVM** (écart de F1 ≈ 0,005, inférieur à la variation entre plis), mais plus **simple**, **interprétable** et **rapide**, avec des probabilités directes.
 
| Jeu de test | Précision | Rappel | F1 |
|---|---|---|---|
| Bons vins (202) | 0,394 | 0,792 | 0,526 |
| Mauvais vins (862) | 0,936 | 0,715 | 0,811 |
 
Accuracy 0,729 · AUC 0,82.
 
- Le modèle **généralise bien** : F1 de 0,539 en validation croisée contre 0,526 sur le test.
- Il est fiable pour **écarter les mauvais vins** (94 % de bonnes réponses quand il dit « mauvais »), moins pour certifier les bons (39 %).
- **Variables déterminantes** : l'alcool (effet positif le plus fort), la densité et l'acidité volatile (effets négatifs).
**Limites** : note sensorielle subjective, informations absentes (cépage, millésime), seuil ≥ 7 arbitraire.
**Pistes** : ajustement du seuil de décision, méthodes d'ensemble (Random Forest, AdaBoost).
 
Le détail de chaque étape est dans le **rapport technique**.
 

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