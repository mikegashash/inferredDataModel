# XML Insurance Classifier (PyTorch)

Define a custom PyTorch dataset class RichXMLDataset to ingest shredded XML insurance policy data, extracting structured features such as premium amount, age, term length, and risk score. 

Normalize the numerical features using StandardScaler, and encode policy types as class labels using LabelEncoder. 
Use a basic Multi-Layer Perceptron (MLP) model built with nn.Module to classify the policies into multiple categories. 
Load the data with PyTorch DataLoader objects, split into training and validation sets, and train the model over multiple epochs while monitoring classification performance on the validation set.

This project defines a custom PyTorch dataset (`RichXMLDataset`) to parse structured insurance policy data from shredded XML files. 

It extracts key numeric features such as premium amount, insured age, policy term, and risk score. 
These features are normalized and fed into a simple Multi-Layer Perceptron (MLP) model for multi-class classification of policy types (e.g., auto, home, health, life, business). 
The pipeline includes data loading, training, and validation evaluation, using Scikit-Learn metrics for reporting precision, recall, and accuracy.
