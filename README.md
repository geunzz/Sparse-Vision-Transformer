# Sparse-Vision-Transformer
This module contains code files for training and testing S-ViT.  
**1. Sparse_ViT.py** : Sparse Vision Transformer (S-ViT) model class, a class based on ViT with some transformations added. It is possible to receive input properties such as embedding dimension size and patch size and create a model reflecting the corresponding size.  
**2. train_S-ViT** : This is a file that trains the S-ViT model. First, the model description is output so that the number of defined S-ViT parameters can be identified, and then learning is performed for the specified epochs.  
**3. test.py** : A file that tests the trained model. It is possible to select whether closed-set or open-set, identification, and verification, and a test suitable for each can be performed.
