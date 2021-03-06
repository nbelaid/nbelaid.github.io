---
title: Initiation à `R`
subtitle: DU Analyste Big Data
output:
    html_document:
        toc: true
        toc_float: true
        toc_depth: 3
---

Dans ce document sont présentés les concepts principaux du logiciel `R` (éléments de langage, données manipulées, manipulation de données, ...).

## Données et éléments de langage

### Types de données

`R` permet de manipuler différents types de données tel que :

- scalaire : une valeur de type `numeric`, `character`, `logical`, `date`
- `vector` : tableau de scalaires de même type (`numeric`, `character`, `logical`, `date`, `factor`)
- `matrix` : tableau à deux dimensions de scalaires de même type (`numeric`, `logicial` ou `character`)
- `array` : extension de `matrix` à $d$ dimensions
- `data.frame` : ensemble de lignes (entités, parfois nommées) décrites par des colonnes nommées (dites variables aussi), pouvant être de types différents - très proche de la définition d'une table dans un SGBD classique
- `list` : liste d'éléments (nommés ou non, et pas forcément de même type)

Voici un exemple avec le jeu de données `mtcars` déjà présent dans le logiciel `R` :

```{r}
# Type de la variable mtcars
class(mtcars)
# Dimensions
dim(mtcars)
nrow(mtcars)
ncol(mtcars)
# Données en elles-mêmes
mtcars
```

### Typage faible

`R` est un langage de programmation à typage faible : le type de la variable est déterminée par la valeur qui lui est affectée. Ce qui veut dire qu'il est possible d'affecter une chaîne de caractères à une variable numérique. Celle-ci deviendra automatique de type chaîne de caractère. Et il n'y a pas d'opérateurs de déclaration de variables. Celle-ci est créée à sa première utilisation, dans l'environnement actuel.

#### Création d'une variable numérique

```{r}
a = 1
print(a)
class(a)
```

#### Affectation d'une chaîne de caractère à une variable numérique

```{r}
a = "bonjour"
print(a)
class(a)
```

#### Conclusion

**Attention** donc lors de l'écriture des programmes...

### Langage scripté

`R` est un langage scripté : il faut exécuter les commandes les unes après les autres dans la console. Mais il est possible d'écrire des scripts dans des fichiers (souvent avec l'extension `.R`), puis de les appeler via la commande `source()`

```{r, eval=FALSE}
source("chemin/vers/mon/fichier.R")
```

### Accès aux données

Voici quelques éléments pour accéder aux différentes valeurs présentes dans le `data.frame` `mtcars` :

#### Valeurs de la 1ère ligne -> vector

```{r}
mtcars[1,]
```

#### Valeurs de la 1ère colonne (nommée mpg) -> vector

```{r}
mtcars[,1]
mtcars$mpg
mtcars[1] # en mode data.frame
```

#### Valeur à la cellule (1,1)

```{r}
mtcars[1,1]
mtcars$mpg[1]
```

#### Noms des variables

```{r}
names(mtcars)
colnames(mtcars)
```

#### Nom des lignes

```{r}
rownames(mtcars)
```

#### Descriptif de chaque variable d'un `data.frame`

```{r}
str(mtcars)
```

### Eléments de langage

Voici quelques commandes utiles

#### Création d'un vecteur et d'une matrice

```{r}
# vecteur de numériques
c(1, 3, 5)
# vecteur de chaînes de caractères
c("a", "b")
# vecteur de type et de taille définie
vector('logical', 5)
# séquence de base
1:5
# idem
seq(1, 5)
# séquence avec définition de la taille du vecteur
seq(1, 5, length = 10)
# séquence avec définition du pas
seq(0, 1, by = 0.1)
# répétition d'une valeur
rep(1, 5)
# répétition d'un vecteur
rep(1:2, 5)
# répétition d'un vecteur, 2ème version
rep(1:2, each = 5)
# matrice
matrix(1:10, 2, 5)
# création d'un vecteur et redimensionnement de celui-ci -> résultat identique à précédemment
m = 1:10
dim(m) = c(2,5)
print(m)
```

#### Fonction sur les lignes ou les colonnes d'une matrice

Application d'une fonction (`sum()` ici) sur les lignes (`1`) ou sur les colonnes (`2`) d'une matrice.

```{r}
apply(m, 1, sum) # somme sur les lignes
apply(m, 2, sum) # somme sur les colonnes
```

#### Fonction sur les parties d'un vecteur

Application d'une fonction (ici `mean()`) sur les groupes de valeurs d'un vecteur, groupes déterminés par les modalités d'un autre vecteur.

```{r}
# Consommation des voitures (Miles/(US) gallon)
mtcars$mpg
# Transmission (0 = automatic, 1 = manual)
mtcars$am
# Consommation moyenne par type de transmission
tapply(mtcars$mpg, mtcars$am, mean)
```

#### Combinaison des deux fonctions précédentes

En combinant les deux méthodes vu avant, on peut effectuer des opérations de type moyenne de plusieurs variables d'un jeu de données, pour chaque modalité d'une variable qualitative.

```{r}
# Moyenne par type de moteurs (en terme de nombre de cylindres)
# sur les variables quantitatives de mtcars
apply(mtcars[,c("mpg", "disp", "hp", "drat", "wt", "qsec")], 2, tapply, mtcars$cyl, mean)
```


## Interrogation de données

Il est fait référence ici aux principes de l'interrogation de données au sens `SQL` (et plus particulièrement la partie requêtage).

### Restriction et projection

La fonction `subset()` permet d'effectuer les deux opérations. Dans le paramètre `subset`, on indique la condition (simple ou combinée) permet de faire la restriction. Et dans le paramètre `select`, on liste les variables de la projection.

```{r}
subset(mtcars, subset = cyl == 4, select = c(cyl, mpg))
```

Il est aussi possible de faire ces deux opérations dans les `[]` avec un test logique et en indiquant la liste des variables à prendre.

```{r}
mtcars[mtcars$cyl == 4, c("cyl", "mpg")]
```

### Calcul de nouvelles variables

La fonction `tranform()` permet d'ajouter de nouvelles variables, soit résultantes d'un calcul (cf ci-dessous), soit permettant seulement un renommage d'une variable.

```{r}
transform(mtcars[,c("cyl", "disp")], disp_cyl = round(disp / cyl, 2))
```

### Calcul d'agrégat

On peut déjà se servir de la combinaison des fonctions `apply()` et `tapply()` comme vu précédemment. Il existe aussi la fonction `aggregate()`, où l'on définit explicitement les agrégats à effectuer et les calculs à produire (un seul type de calcul par fonction néanmoins).

```{r}
# Agrégat simple : consommation moyenne générale
aggregate(mpg ~ 1, data = mtcars, mean)
# Agrégat classique : consommation moyenne en fonction du nombre de cylindres
aggregate(mpg ~ cyl, data = mtcars, mean)
# Agrégat à deux variables : idem en fonction en plus de la transmission
aggregate(mpg ~ cyl + am, data = mtcars, mean)
# Agrégat sur deux variables : idem mais pour la puissance aussi
aggregate(cbind(mpg, hp) ~ cyl + am, data = mtcars, mean)
```

### Jointures

Pour effectuer tout type de jointure, nous utilisons la fonction `merge()`, dans laquelle on indique les deux tables (correspondant à `x` et `y` respectivement). Les paramètres booléens `all`, `all.x` et `all.y` permettent de dire, s'ils sont à la valeur `TRUE`, quelle type de jointure externe on désire (resp. `FULL`, `LEFT` et `RIGHT`). Par défaut, la jointure se faire sur l'égalité des variables de mêmes noms entre les deux `data.frame`. On peut spécifier les variables à utiliser dans les paramètres `by`, `by.x` et `by.y`.

```{r}
moteur = data.frame(cyl = c(4, 6, 8, 12), def = c("petit moteur", "moteur moyen", "gros moteur", "encore plus gros moteur"))
print(moteur)
merge(
  moteur,
  aggregate(mpg ~ cyl, data = mtcars, mean),
  all = TRUE
)
```

### Sinon, du `SQL` sur les `data.frame`

La librairie `sqldf` permet d'exécuter des requêtes `SQL` directement sur les `data.frame` présents dans `R`. Voici un exemple simple :


## Quelques statistiques descriptives

La fonction `summary()` calcule des statistiques basiques sur un vecteur, celles-ci étant dépendantes du type du vecteur. Si elle est appliquée sur un `data.frame`, elle s'applique sur chaque variable.

```{r}
summary(mtcars)
summary(mtcars$mpg)
summary(mtcars$cyl)
summary(as.factor(mtcars$cyl))
```

### Univarié

On peut accéder aux fonctions de calculs des statistiques descriptives directement. Pour les variables quantitatives, nous allons utiliser comme exemple `mpg` qui représente la consommation.

```{r}
mean(mtcars$mpg)
sum(mtcars$mpg)
var(mtcars$mpg)
sd(mtcars$mpg)
min(mtcars$mpg)
max(mtcars$mpg)
range(mtcars$mpg)
median(mtcars$mpg)
quantile(mtcars$mpg)
quantile(mtcars$mpg, probs = 0.99)
quantile(mtcars$mpg, probs = c(0.01, 0.1, 0.9, 0.99))
```

Il existe tout un ensemble de fonctions graphiques, dont voici quelques exemples.

```{r}
hist(mtcars$mpg)
hist(mtcars$mpg, breaks = 20)
hist(mtcars$mpg, breaks = c(10, 15, 18, 20, 22, 25, 35))
boxplot(mtcars$mpg)
qqnorm(mtcars$mpg)
qqline(mtcars$mpg)
```

Pour les variables qualitatives, nous allons utiliser la variable `cyl` qui représente le nombre de cylindre. Celle-ci étant codée numériquement, toutes les fonctions vues précédemment pour s'appliquer (mais n'avoir aucun sens).

```{r}
table(mtcars$cyl)
prop.table(table(mtcars$cyl))
barplot(table(mtcars$cyl))
pie(table(mtcars$cyl))
```

### Bivarié

#### Quanti - Quanti

Dans ce cadre, on peut bien évidemment calculer les statistiques usuelles (covariance, corrélation) et le nuage de points.

```{r}
cov(mtcars$mpg, mtcars$wt)
cor(mtcars$mpg, mtcars$wt)
plot(mtcars$mpg, mtcars$wt)
plot(mtcars$mpg ~ mtcars$wt)
```

On peut aller plus loin en faisant un modéle linéaire simple.

```{r}
m = lm(mpg ~ wt, data = mtcars)
m
summary(m)
plot(m)
plot(mpg ~ wt, data = mtcars)
abline(m, col = "red")
```

#### Quali - Quali

Ici, on calcule bien évidemment la table de contingence, mais aussi les fréquences totales et marginales (en lignes et en colonnes).

```{r}
t <- table(mtcars$cyl, mtcars$am)
print(t)
prop.table(t)
prop.table(t, margin = 1)
prop.table(t, margin = 2)
mosaicplot(t)
assocplot(t)
barplot(t)
barplot(prop.table(t, margin = 2))
barplot(t, beside = T)
barplot(prop.table(t, margin = 2), beside = T)
```

#### Quali - Quanti

En plus d'obtenir les statistiques par modalité de la variable qualitative, on peut représenter les boîtes à moustaches.

```{r}
tapply(mtcars$mpg, mtcars$cyl, mean)
tapply(mtcars$mpg, mtcars$cyl, summary)
boxplot(mpg ~ cyl, data = mtcars)
par(mfrow = c(3, 1), mar = c(2, 2, 2, 0) + 0.1)
for (c in c(4, 6, 8)) {
  hist(mtcars$mpg[mtcars$cyl == c], xlim = range(mtcars$mpg), main = c)
}
```


## Manipulation de listes

Il existe un type `list` en `R`, permettant de représenter un ensemble d'objets complexes, éventuellement avec des schémas différents et pouvant contenir eux-mêmes toutes sortes d'objets (`vector`, `matrix`, `data.frame`, `list`, ...).

```{r}
l = list(a = "chaîne",
         b = 12,
         c = 1:10,
         d = head(mtcars),
         e = list(x = 1:10, y = log(1:10)))
length(l)
names(l)
l[1]
l[[1]]
l$a
l[1:2]
l[c("a", "c")]
l[sapply(l, length) == 1]
```

### `lapply` et autres

La fonction `lapply()` permet d'exécuter une fonction (passée en deuxième paramètre) à chaque élément d'une liste (passée en premier paramètre), ou un vecteur. Il existe aussi les fonctions `sapply()` et `vapply()`, qui sont similaires mais qui cherchent à simplifier le résultat (la deuxième permet de spécifier le nom des retours de la fonctions, si ceux-ci sont multiples).

```{r}
lapply(l, class)
sapply(l, class)
```


### Fonction particulière

On a parfois (voire souvent) besoin d'utiliser une fonction spécifique dans les fonctions comme `lapply()`. On peut soit la définir avant et l'utiliser comme une autre.

```{r}
infoElement <- function(e) {
    return(c(classe = class(e), longueur = length(e)))
}
lapply(l, infoElement)
sapply(l, infoElement)
vapply(l, infoElement, c(CLASSE = "", LONG = ""))
```

### Fonction anonyme

Mais puisqu'on ne l'utilise généralement que dans cette fonction, il est possible de la déclarer directement dans la fonction `lapply()`. On parle alors de *fonction anonyme* (comme en *JavaScript* par exemple).

```{r}
sapply(l, function(e) {
    return(c(classe = class(e), longueur = length(e)))
})
```

### Fonctionnement *invisible*

On a parfois besoin d'appliquer une fonction qui ne retourne rien à une liste, par exemple pour afficher l'élément ou une partie de celui-ci. Dans l'exemple ci-dessous, on remarque que le code affiche bien chaque élément, mais renvoie aussi une liste contenant les éléments (qui est donc identique à la liste passée en paramètre). Ce comportement est dû au fait que la fonction ne renvoie pas de résultat.

```{r}
lapply(l, function (e) { print(e); })
```

Dans ce type de cas, si on veut éviter ce comportement, on peut utiliser la fonction `invisibile()`. Ceci va rendre invisible l'exécution du code et on ne verra donc pas la liste retournée par `lapply()`.

```{r}
invisible(lapply(l, function (e) { print(e); }))
```


### Fonctions autres

#### Recherche

Il est possible de faire une recherche dans une liste (ou un vecteur) avec les fonctions `Find()` et `Position()`. Celles-ci renvoient le premier élément trouvé (ou le dernier car il est possible de partir de la droite). La fonction passée en premier paramètre doit renvoyer les valeurs `TRUE` ou `FALSE`.

On cherche par exemple ici le premier (ou dernier) élément de type `vector` dans la liste précédemment créée.

```{r}
Find(is.vector, l)
Find(is.vector, l, right = TRUE)
Position(is.vector, l)
Position(is.vector, l, right = TRUE)
```

#### Filtre

Pour récupérer tous les éléments d'une liste respectant une condition (grâce à la fonction passée en paramètre donc), on dispose de la fonction `Filter()`. Nous récupérons ici tous les éléments de la liste qui sont de type `vector`.

```{r}
Filter(is.vector, l)
```

#### Réduction

On peut opérer une opération de réduction d'une liste à l'aide d'une fonction binaire (à deux paramètres donc).
```{r}
Reduce(function(a, b) return(a + b), 1:5, 0)
```

Pour fonctionner correctement, la fonction doit retourner un objet utilisable dans la fonction. Dans l'exemple ci-dessous, nous transformons `mtcars` en une liste de `r nrow(mtcars)` éléments, chacune étant une liste nommée des caractéristiques de la voiture (avec en plus le nom de celle-ci).

```{r}
mt = lapply(1:nrow(mtcars), function(i) {
        return(c(nom = rownames(mtcars)[i], as.list(mtcars[i,])))
    })
unlist(mt[[1]]) # unlist() utilisé pour améliorer l'affichage
```

Imaginons qu'on souhaite faire la somme des consommations, il nous faut créer une liste initiale avec la valeur `0` pour l'élément `mpg`. Ensuite, on additionne les deux valeurs qu'on stocke dans `a` (qui aura pour première valeur `init`) et on retourne celle-ci.

```{r}
init = list(mpg = 0)
Reduce(function(a, b) { a$mpg = a$mpg + b$mpg; return(a)}, mt, init)
```



## A faire

A partir des données présentes dans le fichier [`world-liste.RData`](donnees/World/world-liste.Rdata), répondre aux questions suivantes. Ces données concernent les pays dans le monde (à la fin du siècle dernier), et sont représentées sous forme de liste dans l'objet `Country.liste`.

1. Donner le nombre de pays représentés
1. Trouver les informations concernant la `France`
1. Lister les pays du continent `Antarctica`
1. Calculer la population mondiale
1. Afficher quelques informations pour chaque pays (un pays par ligne) :
    - si l'année d'indépendance (`IndepYear`) est `NA`, alors on affichera

    `pays (continent)`

    - sinon, on affichera :

    `pays (continent - indépendance en XXXX)`
1. Créer une nouvelle liste avec le de ratio de la population des villes du pays (voir `City`) sur la population du pays (`Population`)
1. Identifier (donner leur nom) les pays ayant un ratio supérieur à `1` avec la nouvelle liste créée
