pipenv run python run.py --description $1 --model v2 --form t1 --label outcome --hyperparameters hyperparameters.json --split 341d0e93-c536-4f4b-9178-c70e7b6f9ffd
pipenv run python run.py --description $1 --model v2 --form t2 --label outcome --hyperparameters hyperparameters.json --split 341d0e93-c536-4f4b-9178-c70e7b6f9ffd
pipenv run python run.py --description $1 --model v2 --form features --label outcome --hyperparameters hyperparameters.json --split 341d0e93-c536-4f4b-9178-c70e7b6f9ffd
bash notify.sh "renal-mri all trials complete for $1"
