# Challenge_inf554


idées:
    - il y deux manières d'influer sur le score: 
            - le preprocessing + embedding
                => c'est pas sur que ce soit une bonne chose d'enlever les majuscules ou la ponctuation: on pourrait faire une colonne qui compte les "!" ou les majuscules, elles seraient assez représentatives d'un évènement majeur dans le match

                => D'abord on s'entraine que sur une petite partie du dataset total: j'ai créé un sub_train_tweets
                   Puis on va garder en mémoire les preprocessing car ils seront très lourds à tourner

            - le modèle
                => on peut tester des modèles très simples de machine learning à partir de labels hyper explicites et logiques par rapport à la prédictions. exemple: si il y a goal, but dans les messages etc.
                => on peut faire des trucs plus sous le tapis ou on construit un neural network qui fait tout



questions:
    - ici le contexte aide beaucoup à trancher: tous les éléments de la minute ont la même classe donc croiser les messages aide beaucoup. On peut s'attendre à de meilleurs résultats si on croise une ligne avec les autres de sa minute voire plus que si on doit classer uniquement basé sur le message de la ligne... => dans le preprocessing, un vecteur = 1 minute au final??
        => oui car à la fin il faut rendre un csv avec pour chaque minute une classe 0 ou 1.

    -

vocabulaires et notions:

    - NLP: traitement du langage naturel

    - LLM: large language model

    - GloVe: technique d'embedding (=représentation vectorielle), on part d'une matrice de co-ocurrence (matrice d'adjacence d'un graphe ou un mot est un noeud et le poids de chaque connexion entre les noeuds est le nombre de fois que les mots apparaissent à côté, dans la même phrase), le but est ensuite de factoriser cette matrice pour exprimer chaque mot dans une base de dimension plus petite (beaucoup plus petite que la taille du vocabulaire). 
        => script:
            model = api.load("glove-twitter-200")
            words = tweet.split()  
            word_vectors = [model[word] for word in words if word in model]
            ici le model est une base de donnée qui à chaque mot attribue un vecteur de dimension 200 (la base est bien des mots de références)
            le model permettrait aussi de mesurer la similarité avec des mots: model.similarity('king', 'queen')

    - lemmatisation: transformer tous les mots d'une même famille en un seul mot clef. ex: better -> good, cats -> cat. WordNetLemmatizer est une fonction de nltk qui permet de faire ça.

    -dummy classifier:
        Un dummy classifier se contente d'appliquer une stratégie pré-définie sans tenir compte des données d'entrée (ou avec une utilisation très limitée des données). Voici les stratégies courantes :

        most_frequent :
        Prédit toujours la classe majoritaire (celle qui apparaît le plus souvent dans les données d'entraînement).
        Par exemple, si dans un jeu de données, 70% des échantillons appartiennent à la classe "A", le modèle prédit toujours "A".

        stratified :
        Prédit une classe au hasard, mais en respectant les proportions des classes observées dans les données d'entraînement.
        Par exemple, si 70% des échantillons appartiennent à la classe "A" et 30% à la classe "B", il prédit "A" 70% du temps et "B" 30% du temps.

        uniform :
        Prédit une classe de manière purement aléatoire, en supposant que toutes les classes ont la même probabilité.
        Chaque classe a une chance égale d'être prédite.

        constant :
        Prédit toujours une classe fixe définie par l'utilisateur.

    - LSTM: long short term memory, c'est un réseau de neurones plus poussé que le recurrent neural network. Le concept reste le même, prédire non pas sur l'input à t mais sur l'état caché à t: un vecteur de même dimension calculé à partir de l'input à t et de l'état caché à t-1. Dans le cas du LSTM, il y a deux états cachés: long term memory et short term memory. On prédit sur le short term memory calculé à partir de l'input à t et des deux vecteur de memory à t-1.


preprocessings codés:

    1) basic:
    =issu de baseline
        - pour chaque tweet du dataframe: Lowercasing, Remove punctuation, Tokenization, Remove stopwords, Lemmatization
        - calculer le vecteur moyen en glove: pour chaque mot dans le tweet le convertir en vecteur glove puis calculer un vecteur moyen pour l'ensemble du tweet
        - refaire la moyenne ensuite sur la minute avec un group by match et period ID

    2) basic +:
        - la même chose mais on rajoute des colonnes pour compter le nombre de majuscules et de ! (moyenner sur le nombre de mot par tweet et par periode)


modèles codés: 

    1) logistic_regression issu de baseline
        sub_train_test / basic_and_additionnal_preprocessing: accuracy = 0.667 (varie de 0.02)
    2) un neural network avec pytorch complètement au hasard
        sub_train_test / basic_and_additionnal_preprocessing: accuracy = 0.62 (varie de 0.02)

partie technique:

    -si tu veux adapter le linter: Ctrl Shift P, select interpreter et prendre celui qui correspond à là où tu as pip install les packages
    -problèmes d'imports de fonction
        REGLE: on execute des fonctions uniquement à la racine du projet! (les chemins des imports de fichiers pour construire d'autres fonctions sont tous designés pour)
        exemple: from models._Model import Model est une lignes ds tous les fichiers de models. => l'import renvoie une erreur si l'interpréteur n'est pas lancé depuis la racine