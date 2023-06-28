# Notes pour présentation

## Taxonomie des système de recommandation

Il existe de nombreux systèmes de recommandation, qui peuvent être classifiés de différentes manières en fonction de leurs approches sous-jacentes. Voici une taxonomie générale des systèmes de recommandation:

**Système de recommandation non personnalisé**

**Système de recommandation personnalisé**
1. **Systèmes de recommandation basés sur le contenu**: Ces systèmes recommandent des articles en fonction des caractéristiques des articles que l'utilisateur a aimés dans le passé. Par exemple, si vous aimez une recette particulière sur Food.com, le système de recommandation basé sur le contenu pourrait recommander d'autres recettes avec des ingrédients similaires ou du même type de cuisine.
2. **Systèmes de recommandation basés sur la collaboration**: Ces systèmes recommandent des articles en fonction de la **similitude entre les utilisateurs**. Par exemple, si vous et un autre utilisateur avez tous deux aimé certaines recettes, alors le système pourrait vous recommander des recettes que cet autre utilisateur a aimées, mais que vous n'avez pas encore essayées.
   1. **Filtrage collaboratif basé sur la mémoire**: Utilise l'ensemble complet de données pour générer une recommandation. Il utilise des techniques telles que la similitude cosinus, la corrélation de Pearson, etc.
   2. **Filtrage collaboratif basé sur le modèle**: Ce sont des méthodes qui développent un modèle à partir des données des utilisateurs afin de prédire les intérêts de l'utilisateur.
3. **Systèmes de recommandation hybrides**: Ces systèmes combinent à la fois des approches basées sur le **contenu** et des approches basées sur la **collaboration** dans le but de compenser les faiblesses de l'un ou l'autre.
4. **Systèmes de recommandation basés sur le contexte**: Ces systèmes tiennent compte du contexte pour recommander des items. Par exemple, si l'utilisateur cherche des recettes à midi, le système peut recommander des recettes pour le déjeuner.