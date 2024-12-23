# Med-VAE
Conditional variational autoencoder for generation of synthetic tabular medical data.

# Methods

## Dataset Description and Environment

The study utilized data from the SCI_Diabetes dataset, which encompasses comprehensive diabetes care records across Scotland over a five-year period. The dataset comprised 70,162 patient records with 68 features, split into training (n=56,130, 80%) and test (n=14,032, 20%) sets. The implementation was conducted using Python 3.10.12 on a Linux 5.19.0-32-generic system (x86_64 architecture). Code and implementation details are available at https://github.com/hogglet-rsc/Med-VAE.

## Data Preparation 

Patient data was imported and preprocessed to ensure consistent formatting across all features. Categorical variables (foot risk, eye risk, diabetes type, smoking status, and ethnicity) were encoded from one-hot encoding to ordinal values using OrdinalEncoder. Binary features, initially stored as boolean data types, were converted to integers. To maintain dataset quality, drugs prescribed at a rate of less than 10% were excluded from the analysis.

Missing values were handled using multiple imputation with chained equations (MICE), which was selected after comparative testing against simpler approaches like median imputation with and without imputation flags. For the foot_risk assessment scale, we standardized the highest risk category by remapping value '4' to '3', creating a final four-point scale (0-3).

The dataset underwent thorough quality control, including scanning for infinite values, NaNs, and non-numeric columns. Gradient columns were removed due to noise from frequency-based measurements. We performed an 80/20 train-test split, and numerical features were scaled using MinMaxScaler. To verify data integrity post-processing, descriptive statistics (mean, standard deviation, kurtosis, and skewness) were calculated after scaling to ensure the data distribution remained unchanged.

The processed data was organized into separate dictionaries for training and test sets, with features categorized as numerical, binary, or categorical (foot_risk, eye_risk, dm_type, smoking, and ethCode).

## Model Architecture and Development 

Med-VAE employs a variational autoencoder architecture optimized for medical data synthesis. The model comprises three main components:

1. An encoder network that compresses patient information into a probabilistic latent representation
2. A sampling mechanism that operates in this latent space
3. A decoder network that reconstructs synthetic patient records

The encoder processes categorical features through learned embeddings (three dimensions per category) before combining them with numerical and binary inputs. This combined input passes through three neural network layers (88, 60, and 32 neurons respectively) with leaky ReLU activation functions, progressively compressing the information. Each layer incorporates batch normalization, which was retained after showing consistent performance improvements in our architecture experiments. Other tested modifications, including 10% dropout rates and skip connections, did not yield reliable benefits.

The encoder outputs parameters (mean and log variance) for a 32-dimensional latent space, where each dimension approximates a standard normal distribution. To ensure this normal distribution is achieved, we employ KL divergence in the loss function with an annealing schedule. The annealing factor scales linearly from 0 to 1 across training epochs, allowing the model to initially focus on reconstruction accuracy before gradually enforcing the distribution constraint. This approach helps prevent the "posterior collapse" problem common in VAE training.

The decoder mirrors the encoder's architecture in reverse, using three layers (32, 60, and 88 neurons) with leaky ReLU activations and batch normalization. The final layer uses appropriate activation functions for each feature type:
- Sigmoid for binary features
- Softmax for categorical features
- Linear activation for continuous variables

## Training and Validation Framework 

The model was optimized using the Adam optimizer with a learning rate of 0.001, determined through extensive testing across a range of 0.0001 to 0.05. The loss function combined seven distinct reconstruction losses:
- Mean squared error for numerical features
- Binary cross-entropy for binary features
- Sparse categorical cross-entropy for categorical features

To balance these multiple objectives, loss weights were calibrated with a weight of 100 for numerical features, 1 for categorical features, and an optimized binary_weight for binary features to ensure comparable initial loss magnitudes across terms.

Training performance was monitored through both component-wise and aggregate metrics. The model's convergence was assessed at various epochs (100, 200, and 300), with optimal performance identified at 200 epochs. Beyond this point, increased training time led to higher variance in outcomes without consistent improvement in synthetic data quality.

To ensure robust evaluation, we established a comprehensive validation framework comparing Med-VAE against three leading synthetic data generators from the SDV framework: CT-GAN, Gaussian Copula, and T-VAE. For fair comparison, all models underwent identical data preparation steps and were trained on the same hardware. Neural network-based models (Med-VAE, CT-GAN, T-VAE) were each trained four times at different epoch settings, while the non-neural Gaussian Copula model was run five times to establish confidence intervals. All SDV models used factory settings to ensure reproducibility.

Performance evaluation employed two complementary approaches:

1. The SDV quality assessment framework, which combines column pair trend matching and column shape matching to quantify how well synthetic data preserves statistical relationships present in the original data, producing a score between 0 and 1.

2. A practical utility test using a binary classification task: predicting whether a patient's age was over 60. This served as a proxy for real-world medical prediction tasks, such as predicting adverse events in a follow-up interval.

For the classification task, XGBoost classifiers with default hyperparameters were trained on 8,000 synthetic samples generated by each model and tested on 2,000 real patient records from a held-out test set. A baseline classifier trained on 8,000 real data points and evaluated on the same test set provided a performance ceiling for comparison. This approach allowed us to quantify each model's ability to generate synthetic data that maintains the predictive relationships present in real medical records.

## Synthetic Data Generation Process 

Data generation in Med-VAE involves a systematic process of sampling from condition-specific latent space representations. For each medical condition or parameter value of interest (e.g., Type 2 Diabetes), the model first creates a specialized library of latent space tensors through a multi-step process:

1. Training set data points are filtered to isolate all cases matching the condition of interest
2. These filtered data points are then processed through the trained encoder to obtain their latent space representations
3. For each data point, this produces a 32-dimensional tensor where each dimension is characterized by its mean and log variance
4. These tensors are collected and stored as a condition-specific library, maintaining the full distributional characteristics of the original condition-specific cohort

Synthetic data generation begins with random sampling from these stored tensor libraries. For each synthetic case, a tensor is randomly selected from the appropriate condition-specific library. This tensor provides the statistical parameters (mean and log variance) for each of the 32 latent dimensions. The model then generates new synthetic points by sampling from independent Gaussian distributions defined by these parameters. This sampling process leverages the VAE's trained latent space structure, where the KL divergence loss term during training has encouraged each dimension to approximate a standard normal distribution.

The sampled latent vector is then processed through the decoder network, which reconstructs it into a complete synthetic patient record. Continuous variables are produced directly from the decoder's linear output layer, while binary and categorical variables are derived from their respective sigmoid and softmax outputs. This ensures that each synthetic record maintains appropriate value ranges and categorical distributions.

This approach allows for targeted generation of synthetic medical data while preserving both global data distributions and condition-specific characteristics. The process can be applied across various medical parameters, though performance varies with the frequency of conditions in the training data.
