**Date: 15 juin to 18 juin 2026**

**Objectif du jour: Finir la comparaison entre mon code Python et le code Matlab**

Ce qui fonctionne: Ceci va être long car je commences à faire mon daily log.

<ol>
  <li>Lecture vidéo/frames/ROI : Le pipeline Python charge un video, 
       utilise ROI et traite les frames</li>
  <li> Masques fascicules/aponeuroses : Fonctionne très bien </li>
  <li> get_mask/logique de masque MatLab : fonctionnel depuis le 18 juin. 
       Il y avait un problème dans la reconstruction des masques qui s'éloignait 
de ce que MatLab avait.</li>
  <li>Détection Hough sur frames isolées, fonctionne bien. Testées sur certain frames 
problématique. Mais si je reconstruies pareil comme MatLab le fait, la parité est devient 
bonne</li>
  <li>TimTrack image par image : fonctionne bien. Si je fais : "prends un frame et détecte les features
c'est très proche de MatLab</li>
  <li>KLT one-step marche presque parfaitement. En partant de la geométrie Matlab
au frame précédent et prédit le frame après, on est super proche de ce que MatLab fait.
OpenCV/KLT semble donc adéquat</li>
  <li>Kalman 2-State fonctionne si on donne le Kalman avec les bons signaux MAtlab/Oracle.</li>
  <li>Le pipeline avec inputs oracle/matlab marche si on utilise les bons alpha/aponeuroses venant de MATLAB, 
le pipeline Python peut sortir des résultats proches.</li>
</ol>

Différence Python vs MATLAB

| Niveau                | MATLAB UltraTimTrack                                                   | Python actuel                                                                  | Différence principale                                                                                  |
| --------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| Pipeline global       | Pipeline original complet                                              | Réimplémentation en validation                                                 | Python est proche par blocs, mais pas encore fermé en pipeline autonome complet.                       |
| Masques               | `get_fasMask.m`, logique interne MATLAB                                | `fascicle_ellipse_mask`, reconstruction MATLAB-like                            | Python peut reproduire les masques MATLAB si la même logique est appliquée.                            |
| Hough / TimTrack      | `doHough`, `filter_usimage`, `get_fasMask`, extraction des géofeatures | `dohough`, `DoHoughParams`, fonctions dans `ultrasound_tracker.timtrack_hough` | Les résultats sont proches sur les frames testées, mais la séquence complète doit encore être validée. |
| Alpha pleine séquence | Stream `geofeatures.alphas/ws` utilisé par MATLAB                      | Alpha brut Python venant du pipeline TimTrack                                  | L’alpha Python pleine séquence ne reproduit pas encore parfaitement l’alpha MATLAB.                    |
| KLT / UltraTrack      | Tracking MATLAB séquentiel                                             | Tracking Python avec OpenCV / `cv2`                                            | Le KLT Python est bon en one-step, mais drift en mode séquentiel.                                      |
| Kalman                | `do_state_estimation`, modèle 2-state MATLAB                           | `run_matlab_2state_kalman`                                                     | Le Kalman Python fonctionne bien si les entrées alpha/KLT/aponeuroses sont correctes.                  |
| Sortie finale         | `Fdat.Region.FL`, `PEN`, `ANG`                                         | Sorties Python estimées                                                        | La sortie finale Python n’est pas encore fiable sans inputs corrigés ou oracle MATLAB.                 |


| Notebook | Rôle rapide                                                                                                                                            |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| NB20     | Debug de l’alpha Python actuel : vérifie si l’erreur vient seulement d’un signe/angle ou si l’estimateur est différent de MATLAB.                      |
| NB21     | Essaie un pipeline alpha/masque plus “MATLAB-style” avec `dohough + weighted_median`.                                                                  |
| NB22     | Sert de pont vers une correction de production : remplacer l’estimateur d’angle basé sur un segment par `dohough` MATLAB-like.                         |
| NB23     | Lance une séquence TimTrack avec alpha calculé par `dohough + weighted_median` au lieu de l’ancien angle de segment.                                   |
| NB24     | Teste la vraie formule de longueur fasciculaire : `faslen = thickness / sin(phi)` et isole l’effet de `alpha`, `betha` et `thickness`.                 |
| NB25     | Diagnostique pourquoi l’alpha Python `dohough` reste différent de MATLAB : pics Hough, masques, weighted median, résolution angulaire.                 |
| NB26     | Analyse plus finement les `houghpeaks`, la résolution `theta`, les poids des pics et le comportement du `weighted_median`.                             |
| NB27     | Checkpoint visuel rapide : compare les courbes angle/longueur Python vs MATLAB pour voir si le signal est utilisable.                                  |
| NB28     | Compare les sorties finales candidates Python vs MATLAB en valeurs brutes et après normalisation au baseline.                                          |
| NB29     | Déplace le calcul final `ANG/PEN/FL` dans le package avec `ultrasound_tracker.final_output` au lieu de garder la formule seulement dans les notebooks. |
| NB30     | Prototype de Kalman avec mesure adaptative basée sur la cohérence des speckles : l’idée est d’adapter le bruit de mesure selon la confiance image.     |
| NB31     | Prototype de mécanique locale probabiliste : déplacement speckle/block-matching, strain, strain rate, confiance et warnings.                           |
| NB32     | Gate finale de parité MATLAB : compare Python final output avec `Fdat.Region.ANG`, `PEN`, `FL`.                                                        |
| NB33     | Nouvelle gate de parité MATLAB “fresh” : relance ou reprend la comparaison finale avec un contexte plus propre/récent.                                 |
| NB34     | Test Frangi avec seuil adaptatif pour le masque fascicule, probablement pour améliorer la parité du masque avec MATLAB.                                |
| NB35     | Analyse du comportement de `filter_usimage.m` de MATLAB pour comprendre la filtration/thresholding upstream.                                           |
| NB36     | Vérifie la parité des masques intermédiaires MATLAB vs Python, étape par étape.                                                                        |
| NB37     | Calibration du seuil adaptatif pour essayer de reproduire plus fidèlement le masque MATLAB.                                                            |
| NB38     | Test spécifique du comportement `filter_usimage` avec filtre/moyenne 71 et seuil, probablement pour reproduire un détail MATLAB.                       |
| NB39     | Microscope frame par frame du pipeline de masque : regarde précisément où Python diverge de MATLAB dans la construction du mask.                       |


| Notebook | Rôle principal                                                      | Conclusion                                                                                        |
| -------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| NB40     | Validation de `dohough` avec les mêmes masques/internals que MATLAB | `dohough` peut se rapprocher de MATLAB si les entrées sont identiques.                            |
| NB41     | Correction / reconstruction de `get_fasMask`                        | Le masque fascicule Python peut devenir presque identique au masque MATLAB.                       |
| NB42     | Gate TimTrack full pipeline                                         | La partie TimTrack image par image est assez proche pour passer à KLT/Kalman.                     |
| NB43     | Gates downstream KLT/Kalman                                         | TimTrack est proche, mais KLT/Kalman end-to-end ne sont pas encore prêts.                         |
| NB44     | KLT avec masques oracle                                             | Même avec des masques MATLAB/oracle, le KLT séquentiel Python n’est pas encore parfait.           |
| NB45     | Diagnostic KLT one-step affine                                      | Le KLT local frame `f-1 → f` fonctionne bien.                                                     |
| NB46     | KLT drift vs Kalman boundary                                        | La dérive vient surtout du KLT séquentiel, pas du Kalman.                                         |
| NB47     | Variantes de refresh / correction KLT                               | Tentatives pour réduire la dérive séquentielle KLT.                                               |
| NB48     | Port du Kalman MATLAB 2-state                                       | Le Kalman Python fonctionne si les bons inputs sont fournis.                                      |
| NB49     | Scaffold end-to-end corrigé                                         | Le pipeline peut marcher avec handoff corrigé, mais il utilise encore des éléments MATLAB/oracle. |
| NB50     | Fix alpha + KLT drift pipeline gate                                 | Alpha corrigé + KLT one-step + Kalman 2-state donne une sortie proche de MATLAB.                  |


Conclusion

Le code Python est proche de MATLAB pour plusieurs blocs isolés : masques, Hough, TimTrack local, KLT one-step et Kalman 2-state. Le problème restant n’est pas que tout est faux, mais plutôt que le pipeline complet n’est pas encore autonome.

Les deux différences principales sont :

1. l’alpha Python pleine séquence ne reproduit pas encore parfaitement l’alpha MATLAB;
2. le KLT Python séquentiel accumule une dérive que le KLT one-step ne montre pas.

Donc le projet ne doit pas être recommencé. Il faut surtout fermer les deux derniers écarts : le stream alpha complet et le comportement temporel du KLT/UltraTrack.





Bref, ce qui est bon: mask MATLAB-like → doHough → TimTrack local → KLT one-step → Kalman 2-state


Prochaine étape: Dès que j'ai une bonne parité entre les deux codes, je vais faire 
des tests sur les videos complets et faire des comparaisons quantitatives. 
Je vais aussi faire des tests de robustesse en introduisant du bruit dans les frames et voir 
comment les deux codes se comportent.

Figure produite: Notebooks 30 to 54

Question pour superviseur: N/A 

Date: 19 june 

Objectif du jour: Avoir un plan de match en termes d'étapes concrètes.
Si je suis chanceux, finir la parité entre les deux codes.

Ce qui fonctionne: 
Ce qui ne fonctionne pas:
Différence Python vs MATLAB:
Prochaine étape:
Figure produite:
Question pour superviseur:



Gabarit: 
Date:
Objectif du jour:
Ce qui fonctionne:
Ce qui ne fonctionne pas:
Différence Python vs MATLAB:
Prochaine étape:
Figure produite:
Question pour superviseur: