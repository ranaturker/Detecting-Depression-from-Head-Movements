The aim of the study, titled “Depression Detection in Videos from Head Movements”, is to
combine computer vision and machine learning techniques to provide a scalable, effective and
unbiased method for detecting depression. Approximately 280 million people worldwide suffer
from depression, a mental health condition that accounts for more than 60% of suicides. Modern
diagnostic techniques, despite being widely used, are often arbitrary, difficult and error-prone.
This project aims to address these limitations by presenting a novel approach that combines visual
and auditory modalities to effectively and efficiently identify depressive behaviors.
This phase of the project introduces the application of Support Vector Machines (SVM) as a
classification model. In the Training phase of the project, where we have a total of 189 datasets,
the y_matrix.csv dataset consisting of 142 indices (42 depressed and 100 non-depressed samples)
was used. The data was split into two parts, 70% for training and 30% for testing, by applying the
random_state function to ensure consistency across replicates. To further validate the performance
of the model and reduce bias, Leave-One-Out Cross Validation (LOOCV) was used. This
approach maximizes the utility of the dataset by ensuring that each sample is used for both
training and testing, providing a robust performance evaluation.
