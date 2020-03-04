pipenv install --skip-lock

pipenv run python setup.py

export FLASK_APP=api.py
pipenv run flask db init
pipenv run flask db migrate
pipenv run flask db upgrade
