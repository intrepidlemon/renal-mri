pipenv run python calculate_features.py

pipenv run python preprocess.py --n4

bash notify.sh "renal-mri preprocess complete"

bash run-model.sh $1
