# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* **Developed by:** Christina Jernberg
* **Model Type:** Random Forest Classifier (trained using scikit-learn)
* **Version:** 1.0.0
* **Date:** August 2026

## Intended Use
* **Primary Intended Use:** This model is designed to predict whether an individual's income exceeds $50K/year based on demographic and employment data. 
* **Primary Intended Users:** Students and reviewers of the Udacity Machine Learning DevOps Engineer Nanodegree.
* **Out-of-Scope Use Cases:** This model is strictly for educational purposes and should not be used for real-world financial, lending, or employment decisions.

## Training Data
* **Source:** 1994 Census Bureau database (often referred to as the "Adult" dataset).
* **Target:** `salary` (<=50K or >50K).
* **Features:** Demographic and employment attributes including age, workclass, education, marital status, occupation, relationship, race, sex, hours-per-week, and native-country.
* **Preprocessing:** Categorical features were one-hot encoded and the target variable was binarized using a LabelBinarizer. The data was typically split using an 80/20 train-test split.

## Evaluation Data
* The model was evaluated on the remaining 20% test split from the original Census dataset. 

## Metrics
The model was evaluated using standard classification metrics. The final performance on the test set was:
* **Precision:** 0.7391
* **Recall:** 0.6384
* **F1 Score:** 0.6851

*(Additionally, performance slices based on categorical features can be found in `slice_output.txt`)*

## Ethical Considerations
* **Bias and Fairness:** The dataset includes sensitive attributes such as `race` and `sex`. Because the data reflects historical societal structures from 1994, the model is highly susceptible to learning and perpetuating historical biases regarding gender and racial income disparities. 
* **Privacy:** The data is a public, anonymized dataset from the Census Bureau. 

## Caveats and Recommendations
* **Outdated Data:** The data was collected in 1994. Economic conditions, inflation, and demographics have shifted significantly since then, rendering the $50K threshold historically specific and not applicable to modern populations.
* **Recommendation:** If used for anything beyond a technical exercise, the model's fairness across different demographic slices (particularly `race` and `sex`) must be rigorously audited.

