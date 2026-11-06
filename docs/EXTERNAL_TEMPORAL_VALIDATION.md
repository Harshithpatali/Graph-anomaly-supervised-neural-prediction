# Independent later-period validation

## Current evidence boundary

The repository's e-commerce `Fraud_Data.csv` ends on 2015-12-16. Its final chronological holdout is later than its training data, but it is not an independent dataset and must not be described as external validation.

## Candidate independent benchmark: IEEE-CIS Fraud Detection

The IEEE-CIS competition contains labeled online transactions and a relative-time field (`TransactionDT`). It is a later benchmark than the repository's 2015 data and is relevant to online transaction fraud, but it is not a row-compatible continuation of `Fraud_Data.csv`. It must be evaluated as a separate dataset with its own chronological train/validation/test split, not scored by the e-commerce model trained on the original columns.

Dataset page: https://www.kaggle.com/c/ieee-fraud-detection/data

Access requires a Kaggle account and acceptance of the competition rules. The repository does not contain the raw IEEE-CIS files or Kaggle credentials. Do not commit raw competition data or credentials.

## Enable the external benchmark

1. Sign in to Kaggle and accept the IEEE-CIS competition rules.
2. Create a Kaggle API token.
3. Add repository Actions secrets named `KAGGLE_USERNAME` and `KAGGLE_KEY`.
4. Run the external benchmark workflow after its download step is enabled.

The benchmark must preserve `TransactionDT` order and report the last chronological partition only once. Results must be labeled cross-dataset benchmark results, not direct transfer performance on the original e-commerce model.

## Why not use every newer public fraud CSV?

Some newer public demonstration datasets contain labels assigned from a model's own block/allow decisions. Such verification bias can mark unreviewed missed fraud as legitimate. They are not acceptable as clean independent ground truth for model validation.
