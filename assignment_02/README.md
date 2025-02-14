# Classify Candidate Pairs of Acronyms and Expansions (Assignment 2)

The following data sets (dataacro.tar.gz) consist of a training and testing set of candidate pairs of acronyms and expansion from Indonesian text, read this paper for detail. Some examples are:

BMKG=>Badan Metorologi Klimatologi dan Geofisika 1 1:1 2:1 3:1 4:1 5:0.8 6:1 7:1 8:0.97
Bendum=>Bendahara Umum 1 1:1 2:1 3:0.7 4:1 5:1 6:1 7:1 8:0.96
UKM=>Usaha Kecil Menengah 1 1:1 2:1 3:1 4:1 5:1 6:1 7:1 8:1
Bosscha=>Jl Pejambon -1 1:0.92 2:1 3:-1.29 4:0 5:1 6:0.29 7:0 8:0.27
Jl=>Menkom Info -1 1:1 2:1 3:-2 4:0 5:1 6:0 7:0 8:0.14
Mendikbud=>Menteri Pendidikan -1 1:0.97 2:1 3:0 4:0.5 5:1 6:0.67 7:0 8:0.59
BMKG=>Badan Metorologi Klimatologi dan Geofisika is the candidate pair. The number 1 indicates the positive class, the number -1 means the negative class label, and the rest are the features, i.e., 8:0.97 means the 8th feature, whereas 0.97 is the value of that 8th feature. Please read this article for more information about the features.

You are asked to train and find the best model using Support Vector Machine, K-Nearest Neighbours, Naive Bayesian, and Decision Tree. You also have to compare the classification results with BERT (read this paper Pre-training of Deep Bidirectional Transformers for Language Understanding) and learn BERT code at https://github.com/google-research/bert.  

Compare the training and testing results based on the F1-score. Please show the TP, FP, FN, TN, Precision, and Recall. Write a report for this assignment in a PDF format with a cover containing your name and student ID, which includes the following sections, and show some figures or images of your process:

Introduction
Methodology
Results and Discussion
Conclusion
References
Please submit this assignment at the latest on February 8, 2025, at 10.30 am, via our e-learning system. 