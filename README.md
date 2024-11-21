# Challenge_inf554


idées:
    - il y deux manières d'influer sur le score: 
            - le preprocessing
                => c'est pas sur que ce soit une bonne chose d'enlever les majuscules ou la ponctuation: on pourrait faire une colonne qui compte les "!" ou les majuscules, elles seraient assez représentatives d'un évènement majeur dans le match

                => D'abord on s'entraine que sur une petite partie du dataset total puis on va garder en mémoire les preprocessing car ils seront très lourds à tourner

            - le modèle
                => on peut tester des modèles très simples de machine learning à partir de labels hyper explicites et logiques par rapport à la prédictions. exemple: si il y a goal, but dans les messages etc.
                => on peut faire des trucs plus sous le tapis ou on construit un neural network qui fait tout

questions:
    - ici le contexte aide beaucoup à trancher: tous les éléments de la minute ont la même classe donc croiser les messages aide beaucoup. On peut s'attendre à de meilleurs résultats si on croise une ligne avec les autres de sa minute voire plus que si on doit classer uniquement basé sur le message de la ligne... => dans le preprocessing, un vecteur = 1 minute au final??

    -

vocabulaires et notions:

    - NLP: traitement du langage naturel
    - GloVe: technique d'embedding (=représentation vectorielle), on part d'une matrice de co-ocurrence (matrice d'adjacence d'un graphe ou un mot est un noeud et le poids de chaque connexion entre les noeuds est le nombre de fois que les mots apparaissent à côté, dans la même phrase), le but est ensuite de factoriser cette matrice pour exprimer chaque mot dans une base de dimension plus petite (beaucoup plus petite que la taille du vocabulaire). 
    - lemmatisation: transformer tous les mots d'une même famille en un seul mot clef. ex: better -> good, cats -> cat. WordNetLemmatizer est une fonction de nltk qui permet de faire ça.
