petit fichier pour stocker les résultats de nos tests

- le meilleur: feed_forward_neural_net.py avec base (hidden layer: 128 neurones, 100 epochs et 32 de batch size)
    => 0.78 accuracy sur nos matchs (avec du shuffle)
    => 0.67 sur le submit

    bcp d'overfit, le dropout baisse l'accuracy sur nos matchs mais aussi pour le submit
        => # il faut voir si notre test est pertinent: il faudrait peut-être plutôt entrainer sur 10 matchs de train.csv et garder les 2 ou 3 derniers pour tester

- logistic regression + very_simple embedding: 
    => rien de fou, 0.62 
    => il semble tout classifier à 1

- feed_forward + concat de base et very_simple comme embedding:
    => pas fou non plus, 0.63
    
    étonnant car l'info sur le nombre de tweet, les points d'exclamation etc me semblait pertinente. 
        => pb de normaliser les vecteurs, à voir si cela améliore le tout
            => on fera attention de normaliser à chaque fois, pour autant ça fait tjrs quelque chose de pas fou (0.63)
