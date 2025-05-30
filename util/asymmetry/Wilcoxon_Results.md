__Multiple Lesions and the Impact on Technical Implementation of Asymmetry:__


Many asymmetry algorithms in dermoscopic image analysis rely on the lesion being centered within the image to enable symmetry assessment along a fixed axis or axes [1]. This approach was initially implemented in Asym_Comp&ABR_Generator.ipynb. In this method for every image, the number of lesions (components) was computed. Each lesion in the image was then recentered in its own binary image then axis-based reflections were performed to assess asymmetry. The individual asymmetry scores of each lesion in the image were then summed and averaged to give a final asymmetry score for the image. This aligns with standard practices in dermoscopy literature [2]. See appendix A for an example.
Additionally, we introduced a relative bias weight scoring, inspired by the observation that a cluster of components localized within a small spatial range may indicate irregular growth. To evaluate the features, a model was trained independently within our model framework. The seed code of 413316891 was used to evaluate the efficacy of the features.

![image1](https://github.com/BossThePro/2025-FYP-groupKangaroo/blob/main/data/asymmetry/Wilcoxon%20Images/Screenshot%202025-05-27%20143835.png)

Fig 1.1 Fold 5 from seed code 413316891 evaluating the features developed from Asym_Comp&ABR_Generator.ipynb

Given how poorly asymmetry performed in the model from this method, we decided on a far more naive approach. Rather than focusing on centroid and lesions themselves, asymmetry would now operate on the skin itself i.e. the whole image. The scope shifted from discriminating individual lesions asymmetry and averaging; to is the skin of this region asymmetrical. An extremely naive model not supported by literature. With the same seed number of 413316891 the results performed significantly better on the same instance of data.

![image2]https://github.com/BossThePro/2025-FYP-groupKangaroo/blob/main/data/asymmetry/Wilcoxon%20Images/Screenshot%202025-05-27%20143744.png

Fig 1.2 Fold 5 from seed code 413316891 evaluating the features developed from Naive_Asym_Generator.ipynb

To test the significance of the features increased performance a Wilcoxon Test was implemented with a hypothesis test of: 
H_0 = Model A (Asym_Comp&ABR_Generator.ipynb)  = Model B (Naive_Asym_Generator.ipynb)
H_A = Model A (Asym_Comp&ABR_Generator.ipynb) < Model B (Naive_Asym_Generator.ipynb)
Across all 8 folds we saw statistical significance for all at alpha 0.05. Even more so there is strong statistical significance for Accuracy, Recall and F1 Score at alpha 0.01. Thus the features are statistically more significant in diagnosing cancer. 

![image3](https://github.com/BossThePro/2025-FYP-groupKangaroo/blob/main/data/asymmetry/Wilcoxon%20Images/Screenshot%202025-05-27%20153403.png)

Figure 1.3 Showing the results of the Wilcoxon Test

While performing better the multicollinearity that exists in this model should be viewed with scepticism with a 0.9, Pearson's R relationship between ASI and asymmetry score. Essentially because they are computed in nearly the same method, an error conducted in methods that should be avoided in future.
With multiple lesions, the definition of what is considered irregular becomes ambiguous. The sensitivity of asymmetry alone is not high enough for diagnosis, some melanomas can be symmetrical, and some benign lesions can be asymmetrical [3]. This is reflected in our data set with Figure 1.4 being a perfect example. 


![image3](https://github.com/BossThePro/2025-FYP-groupKangaroo/blob/main/data/asymmetry/Wilcoxon%20Images/PAT_87_133_391.png)![image4](https://github.com/BossThePro/2025-FYP-groupKangaroo/blob/main/data/asymmetry/Wilcoxon%20Images/PAT_87_133_391_mask.png)

Figure 1.4 PAT_87_133_391.png showing a benign lesion 

The computed asymmetry of Figure 1.4 is high in both feature extractions. With an overall mean asymmetry score of 0.6380 in Asym_Comp&ABR_Generator.ipynb and a mean asymmetry score of 0.8039 with an ASI of 90.47% Naive_Asym_Generator.ipynb. Thus reinforcing that asymmetry alone is a poor feature without anything to mediate it.
What should be explored in future is how clustering of lesions as well as scale can be utilised as features. This could potentially add more contextual information that helps with asymmetries sensitivities.


References:
1. Clawson, K. M., Morrow, P. J., Scotney, B. W., McKenna, D. J., & Dolan, O. M. (2007). Determination of optimal axes for skin lesion asymmetry quantification. In Proceedings of the 2007 IEEE International Conference on Image Processing (Vol. 2, pp. II-453–II-456)https://www.researchgate.net/publication/4288907_Determination_of_Optimal_Axes_for_Skin_Lesion_Asymmetry_Quantification 

2. Ali, A.-R., Li, J., & O’Shea, S. J. (2020). Towards the automatic detection of skin lesion shape asymmetry, color variegation and diameter in dermoscopic images. PLOS ONE, 15(6), e0234352. https://doi.org/10.1371/journal.pone.0234352

3. Abbasi, N. R., Shaw, H. M., Rigel, D. S., Friedman, R. J., McCarthy, W. H., Osman, I., Kopf, A. W., & Polsky, D. (2004). Early diagnosis of cutaneous melanoma: Revisiting the ABCD criteria. JAMA, 292(22), 2771–2776. https://doi.org/10.1001/jama.292.22.2771
