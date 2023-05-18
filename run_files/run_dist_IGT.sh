#!/bin/bash

L=./lost

# Les parametres de base pour avoir un affichage détaillé et le multi-thread par
# default.

ARGS=" --verbose --nthreads 4"

# On commence par ajouter des patterns pour extraire les features, le format est
# simple :
#	tag:nom:item,item...
# Le tag est optionel et prend la valeur 0 par defaut. Les tags permettent de
# regrouper les features en paquets et de spécifier certains paramètres
# indépendament pour chacun d'entre eux. Le nom permet de garantir que les
# features sont uniques, il sont convertis en un hash comme les items mais
# fixe.
#
# Les items spécifient les morceaux de label à tester dans les features, ils
# sont constitué de trois valeurs : l'arc, le label, et le token.
#   - L'arc peut être 0 ou 1 pour spécifier le premier ou le deuxième arc dans
#     le cas d'une feature bigramme. Si tous les items utilisent le même arc, la
#     feature est unigramme.
#   - Le label peut être 's' ou 't' pour indiquer le label source ou destination
#     de l'arc choisi.
#   - Le token est un entier 0..n qui indique quelle portion de chaine aller
#     chercher. Les différents tokens sont séparés par des '|'.
# Par exemple : 0s0,1t3 indique que la feature test le premier token en source
# du premier arc, et le quatrième token en cible du deuxième arc.
#
# Deux items séparés par un '=' permmettent de faire une feature testant
# l'égalitée de deux tokens : 0s0=0t0 permet de tester que le premier token de
# la source et de la cible sont égaux. Un tel item ne peut prendre que deux
# valeurs, vrai ou faux.
#
# On ajoute ici quelques patrons unigrammes et bigrammes :

# Formerly
# G for gloss, M for source morpheme, W for aligned lemma (word),
# A for alignment index, and L for sentence length.
# Now
# G0 for gloss, B1 for gram_lex, P2 for PoS tag,
# CT3 for target copy, PT4 for target position
# M0 for source morpheme, #W for aligned lemma (word),
# F1 for morpheme position in word, L2 for morpheme length,
# D3 for initial letters, E4 for final characters of the morpheme,
# CS5 for source copy, PS6 for source position.
ARGS+=" --pattern 10:Gx/xx:0t0"
#ARGS+=" --pattern 10:Wx/Wx:0t0,0s0"
#ARGS+=" --pattern 11:Wx/WW:0t0,0s0,1s0"
ARGS+=" --pattern 11:GG/xx:0t0,1t0"
ARGS+=" --pattern 12:Bx/xx:0t1"
ARGS+=" --pattern 13:Px/xx:0t2"
#ARGS+=" --pattern 14:WW/WW:0t0,1t0,0s0,1s0"
ARGS+=" --pattern 15:BB/xx:0t1,1t1"
ARGS+=" --pattern 16:PP/xx:0t2,1t2"

#ARGS+=" --pattern 20:Px/xx:0t1"
ARGS+=" --pattern 20:Gx/Mx:0t0,0s0"
#ARGS+=" --pattern 21:Gx/Wx:0t0,0s1"
ARGS+=" --pattern 22:Gx/Fx:0t0,0s1" #0s2"
ARGS+=" --pattern 23:Gx/Lx:0t0,0s2" #0s3"
#ARGS+=" --pattern 21:Px/PP:0t0,0s1,1s1"
#ARGS+=" --pattern 21:xP/PP:1t0,0s1,1s1"
#ARGS+=" --pattern 22:PP/xx:0t1,1t1"
#ARGS+=" --pattern 23:PP/Px:0t1,1t1,0s1"
#ARGS+=" --pattern 23:PP/xP:0t1,1t1,1s1"
#ARGS+=" --pattern 24:PP/PP:0t1,1t1,0s1,1s1"
ARGS+=" --pattern 24:Gx/Dx:0t0,0s3" # Initial letters
ARGS+=" --pattern 25:Gx/Ex:0t0,0s4" # Final letters

ARGS+=" --pattern 30:GG/Mx:0t0,1t0,0s0"
#ARGS+=" --pattern 31:GG/Wx:0t0,1t0,0s1"
#ARGS+=" --pattern 32:GG/Mx:0t0,1t0,0s2"
#ARGS+=" --pattern 32:GG/Mx:0t0,1t0,0s3"

# For the more complex outputs: Binary category
ARGS+=" --pattern 40:Bx/Mx:0t1,0s0"
#ARGS+=" --pattern 41:Bx/Wx:0t1,0s1"
ARGS+=" --pattern 42:Bx/Fx:0t1,0s1" #0s2"
ARGS+=" --pattern 43:Bx/Lx:0t1,0s2" #0s3" # Morpheme length and binary category
# Experimental bigram
ARGS+=" --pattern 44:BG/xx:0t1,1t0"
ARGS+=" --pattern 45:GB/xx:0t0,1t1"
# With initial and final characters
ARGS+=" --pattern 46:Bx/Dx:0t1,0s3" # Initial letters and binary category
ARGS+=" --pattern 47:Bx/Ex:0t1,0s4" # Final letters and binary category
ARGS+=" --pattern 48:Bx/FF:0t1,0s1,1s1" # Bigram BIOF tag and binary category


# For the PoS tag
ARGS+=" --pattern 50:Px/Mx:0t2,0s0"
#ARGS+=" --pattern 51:Px/Wx:0t2,0s1"
# Experimental bigram
ARGS+=" --pattern 54:GP/xx:0t0,1t2"
ARGS+=" --pattern 55:PG/xx:0t2,1t0"
# With initial and final characters
ARGS+=" --pattern 56:Px/Dx:0t2,0s3" # Initial letters and PoS tag
ARGS+=" --pattern 57:Px/Ex:0t2,0s4" # Final letters and PoS tag

# Copy and dist
ARGS+=" --pattern 60:CTx/xx:0t3"
ARGS+=" --pattern 62:CTx/CSx:0t3,0s5"
ARGS+=" --pattern 65:PTx/PSx:0t4,0s6"
ARGS+=" --pattern 66:GPTx/MPSx:0t0,0t4,0s0,0s6"

#ARGS+=" --pattern 70:Ox/xx:0t5"
#ARGS+=" --pattern 71:Ox/Mx:0t5,0s0"


# Une régularisation l1 et l2 est appliquée avec des paramètres ajustable par
# tag. Tous les tags dont les paramètres ne sont pas spécifiés prennent les
# paramètres du tag 0. (le tag par défaut)
#
# Il est possible de spécifier la fréquence minimum à partir de laquelle une
# feature est intégrée au modèle. (Attention: la feature sera quand même créé
# mais sera ensuite supprimée du modèle, elle va donc quand même occuper de la
# mémoire.) Par défault, la fréquence est le nombre d'apparition dans le
# treillis de recherche, il est possible de changer pour la fréquence dans le
# treillis de référence.
# De même, il est possible de choisir à partir de quelle itération les features
# d'un tag donné seront intégrées au model, et à partir de quelle itération on
# commence à les supprimer si leur score est de zéro.

ARGS+=" --tag-rho1   0:0.5"
ARGS+=" --tag-rho2   0:0.0"
#ARGS+=" --tag-rho3"

#ARGS+=" --min-freq   1:3"
#ARGS+=" --ref-freq   1"
#ARGS+=" --tag-start  1:5"
#ARGS+=" --tag-remove 1:10"

# Puis les données. Pour les données d'entrainement, il faut fournir les fichier
# space qui contiennent les automate représentant les espaces source, ainsi que
# les fichier contenant les transducteur de référence. Pour les deux, les
# options peuvent être répétées s'il y a plusieurs fichiers à fournir.
# Pour le test, il faut fournir le fichier d'espace source ainsi que le nom du
# fichier de sortie à produire.
#
# La présence de fichier de train déclenche l'optimization du modèle et celle de
# test déclenche le décodage après cette éventuelle optimization. Donc, si on
# fourni juste le train, on ne fait que optimizer un modèle, si on fournit juste
# le test, on ne fait que décoder, et si on fournit les deux, les deux étapes
# sont enchainées.
#
# Le fichier de developement, s'il est présent, est décodé à la fin de chaque
# itération. Le numéro de l'itération peut-être inclus dans le nom du fichier de
# sortie avec le format classique de printf.

lang='natugu' #'natugu' #'gitksan' #'lezgi' #'tsez'
suffix='conc_match_dist_791' # 591 #'0_gold_match_1600' # Changer ici
#ARGS+=" --train-spc ../lost_experiment/train_tsez_pos.spc"
#ARGS+=" --train-ref ../lost_experiment/train_tsez_pos.ref"
ARGS+=" --train-spc ../IGT_ST/lost_format/train_${lang}_${suffix}.spc"
ARGS+=" --train-ref ../IGT_ST/lost_format/train_${lang}_${suffix}.ref"

#ARGS+=" --devel-spc dat/devel.spc"
#ARGS+=" --devel-out devel-%02d.txt"
#ARGS+=" --devel-spc ../lost_experiment/dev_tsez_pos.spc"
#ARGS+=" --devel-out ../lost_experiment/devel-pos-%02d.txt"
#ARGS+=" --devel-spc ../results_2023_TALN/data/dev_tsez_${suffix}_pos.spc"
#ARGS+=" --devel-out ../results_2023_TALN/dev_tsez/devel-%02d-${suffix}_pos.txt"
ARGS+=" --devel-spc ../IGT_ST/lost_format/dev_${lang}_${suffix}.spc"
ARGS+=" --devel-out ../IGT_ST/dev/devel-${lang}-%02d-${suffix}.txt"


#ARGS+=" --test-spc  ../lost_experiment/test_tsez_pos.spc"
ARGS+=" --test-spc  ../IGT_ST/lost_format/test_${lang}_${suffix}.spc"
#ARGS+=" --test-out  tsez_exp/output_tsez_pos.out"
ARGS+=" --test-out  ../IGT_ST/output/output_${lang}_IGT_${suffix}.out"
#ARGS+=" --test-fst  tsez_exp/output_tsez_pos.fst"

# Ensuite le modèle. Il est composé de deux fichiers, un contenant les poids des
# features et un contenant les associations entre valeurs de hashage et chaines
# de caractère.
# Par defaut, ce dernier ne contiens que les chaines nécessaires pour que le
# décodeur puisse utiliser le modèle. (voir en dessous)
#
# Il est possible de charger et sauvegarder le modèle. Le chargement ce fait au
# lancement et permet donc soit de charger un modele pour faire du décodage,
# soit pour initialiser un entrainement. La sauvegarde se fait à la fin pour
# enregistrer le modèle entrainé.
# L'option pour compacter demande au systeme de ne sauvegarder que les features
# ayant un poids non-nul. Cela permet de réduire la taille du modèle mais
# surtout d'accéler le décodage.
#
# L'option *-otf permet de sauvegarder le model à chaque iteration pour suivre
# l'évolution des features ou pour choisir le meilleur modèle final. Un flag
# pour un entier au format standard de la fonction printf permet d'ajouter le
# numéro de l'itération au nom du fichier.

#ARGS+=" --mdl-load model.wgh"
#ARGS+=" --str-load model.str"
#ARGS+=" --mdl-load tsez_exp/model_tsez_pos.wgh"
#ARGS+=" --mdl-load ../IGT_ST/dev_model/model-${lang}_IGT_${suffix}-05.wgh"
#ARGS+=" --str-load tsez_exp/model_tsez_pos.str"
#ARGS+=" --mdl-save tsez_exp/model_tsez_pos.wgh"
#ARGS+=" --str-save tsez_exp/model_tsez_pos.str"
ARGS+=" --mdl-save tsez_exp/model_${lang}_IGT_${suffix}.wgh"
ARGS+=" --str-save tsez_exp/model_${lang}_IGT_${suffix}.str"
ARGS+=" --mdl-compact"

#ARGS+=" --mdl-save-otf tmp/model-%02d.wgh"
#ARGS+=" --mdl-save-otf tsez_exp/dev_model_pos/model-pos-%02d.wgh"
ARGS+=" --mdl-save-otf ../IGT_ST/dev_model/model-${lang}_IGT_${suffix}-%02d.wgh"

# L'optimisation est faite avec r-prop avec un nombre fixe d'itérations qu'il
# faut spécifier.  Il est possible aussi d'ajuster les différent paramètres de
# r-prop en cas de besoin. Les paramètres par defaut marchent bien pour la trad
# mais ça peut ne pas être le cas pour d'autres tâches.

ARGS+=" --iterations 15" #15
#ARGS+=" --rbp-stpinc 1.2" #1.2
#ARGS+=" --rbp-stpdec 0.5" #0.5
#ARGS+=" --rbp-stpmin 1e-4"
#ARGS+=" --rbp-stpmax 50"

# Ensuite, on peut regler le niveau de cache, c'est-à-dire la quantité de choses
# gardées en mémoire d'une itération sur l'autre. Ça peut améliorer la vitesse de
# calcul au prix de plus de mémoire :
#    0 -> juste les listes d'arcs avec labels et hypothèses
#    1 -> ajoute les listes de noeud et les connection
#    2 -> ajoute les tris topologique forward et backward
#    3 -> ajoute les listes de features unigrames et bigrammes
#    4 -> ajoute tous les tableaux pour le calcul du gradient
# Jusqu'au niveau 2 la consomation de mémoire reste raisonable mais le gain est
# relativement réduit. Les niveaux 3 et 4 font vraiment exploser la consomation
# de mémoire mais permettent de gagner beaucoup en vitesse si la RAM est très
# rapide. (attention ça peut ralentir aussi si la RAM est lente)

ARGS+=" --cache-lvl 2"

# Et enfin, le debuggage du model. L'option de dump des features crée un
# fichier avec la liste des hash pour chaque features ce qui permet de retrouver
# ce qui l'a généré. Par defaut, seule les chaines nécessaire sont stockées dans
# le modèle, pour être sûr d'avoir toutes les chaines, il faut ajouter la
# deuxième options qui force le stockage de toutes les chaines et permet
# d'utiliser pleinement le dump.
#
# Lorsque le dump est activé, le multi-threading est automatiquement désactivé,
# il est possible de ne faire qu'une seule passe sur le corpus pour créer ce
# fichier et ensuite de relancer l'optimisation sans le dump pour profiter de
# l'acceleration.

#ARGS+=" --ftr-dump model.ftr"
#ARGS+=" --ftr-dump tsez_exp/model_tsez_partial_pos.ftr"
#ARGS+=" --ftr-dump ../results_2023_TALN/model_tsez_${suffix}_pos.ftr"
ARGS+=" --str-all"  # Attention : !BUG! Toujours laisser activé

# Maintenant que l'on a tout, on peut lancer lost :

$L $ARGS
