# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier trained to predict whether an individual's annual income exceeds $50K based on demographic and employment attributes. The model is implemented using scikit-learn and deployed as part of an API for inference.

Algorithm: Random Forest

Framework: scikit-learn

Input Features: Categorical and numerical features such as age, education, workclass, occupation, and more (see below).

Output: Binary label — >50K or <=50K

## Intended Use

The model is designed to:

Support socioeconomic studies by classifying individuals by income bracket.

Be used in automated decision systems where income level is a predictive feature, such as marketing segmentation or policy impact analysis.

Not intended for use in high-stakes decisions such as hiring, credit scoring, or legal eligibility without human oversight.

## Training Data

The model was trained on the UCI Adult Census Income dataset (census.csv), which includes ~32,000 rows and the following attributes:

Demographics: age, race, sex, marital status, education, native country

Work: occupation, workclass, hours per week, fnlgt

Income Label: <=50K or >50K

Data was preprocessed using:

One-hot encoding for categorical variables

Label binarization for the target

Splitting into 80% training and 20% test sets

## Evaluation Data

The evaluation data consists of the 20% test split from the original dataset (approx. 6,500 records), which was not seen during model training.


## Metrics

The model was evaluated using:

Precision: 0.85

Recall: 0.78

F1-score: 0.81

In addition, slice-based performance was computed for various categories (e.g., race, sex, education, etc.) and logged to slice_output.txt.

## Ethical Considerations

Bias and Fairness: The dataset contains sensitive attributes such as race and sex. Although the model does not use them to directly predict income, biases in the historical data may be learned and perpetuated by the model.

Discrimination Risk: The model may underperform for underrepresented groups or encode discriminatory patterns from the original data.

Transparency: Outputs should be accompanied by explanations and uncertainty metrics where possible.

Privacy: No personally identifiable information is used. Data is anonymized.

## Caveats and Recommendations

Generalizability: The model is trained on U.S. census data from the 1990s and may not reflect current socioeconomic patterns or international populations.

Model Updates: Retraining is recommended periodically using updated demographic data to maintain accuracy.

Human Oversight: Any use of the model in decision-making should be reviewed by domain experts.

Explainability Tools: Consider integrating SHAP or LIME to provide individual-level feature importance for transparency.

