import argparse
import os
import math
import json

from datetime import datetime
from models import models
from db import db, Result
from uuid import uuid4, UUID

from keras import backend as K

import numpy as np
import evaluate
from data import data

from config import config

def test_model(model, train, validation, test):

    loss, accuracy = model.evaluate_generator(validation, steps=math.ceil(len(validation)/config.BATCH_SIZE))
    train_loss, train_accuracy = model.evaluate_generator(train, steps=math.ceil(len(train)/config.BATCH_SIZE))
    test_loss, test_accuracy = model.evaluate_generator(test, steps=math.ceil(len(test)/config.BATCH_SIZE))

    # think you mean train here
    train.reset()
    validation.reset()
    test.reset()

    results = evaluate.get_results(model, validation)
    probabilities = list(evaluate.transform_binary_probabilities(results))
    labels = list(evaluate.get_labels(validation))

    test_results = evaluate.get_results(model, test)
    test_probabilities = list(evaluate.transform_binary_probabilities(test_results))
    test_labels = list(evaluate.get_labels(test))

    # think you mean train here
    train.reset()
    validation.reset()
    test.reset()

    return {
        "train_accuracy": float(train_accuracy),
        "train_loss": float(train_loss),
        "accuracy": float(accuracy),
        "loss": float(loss),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "probabilities": probabilities,
        "labels": labels,
        "test_probabilities": test_probabilities,
        "test_labels":test_labels,
    }

def characterize_data(data):
    unique, counts = np.unique(data.classes, return_counts=True)
    index_to_count = dict(zip(unique, counts))
    characterization = { c: index_to_count[data.class_indices[c]] for c in data.class_indices }
    return characterization

def run(model, description, input_form, label_form="outcome", split_id=None, loaded_data=None, hyperparameters=dict()):
    run_id = uuid4()
    if split_id is None:
        split_id = run_id

    history = model.run(run_id, mode='normal', input_form=input_form, loaded_data=loaded_data, label_form=label_form, hyperparameters=hyperparameters)
    K.clear_session()

    model_instance = evaluate.load(os.path.join(
        config.MODEL_DIR,
        "{}-{}.h5".format(str(run_id), model.MODEL_NAME),
        ))

    if loaded_data is None:
        train, validation, test = data(split_id, input_form=input_form, label_form=label_form)
    else:
        train, validation, test = loaded_data
        train.reset()
        validation.reset()
        test.reset()

    train_data_stats = characterize_data(train)
    validation_data_stats = characterize_data(validation)
    test_data_stats = characterize_data(test)
    results = test_model(model_instance, train, validation, test)
    training.reset()
    validation.reset()
    test.reset()

    result = Result(
        model.MODEL_NAME,
        str(run_id),
        str(split_id),
        train_data_stats,
        validation_data_stats,
        test_data_stats,
        description,
        input_form,
        label=label_form,
        hyperparameters=hyperparameters,
        history=history,
        **results
        )
    db.session.add(result)
    db.session.commit()

def explode_parameters(parameters):
    all_parameters = []
    for p in parameters.keys():
        if type(parameters[p]) is list:
            for value in parameters[p]:
                new_parameters = dict(parameters)
                new_parameters[p] = value
                all_parameters += explode_parameters(new_parameters)
            break
    if all_parameters:
        return all_parameters
    return [parameters]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='which model to run (see models.py)')
    parser.add_argument(
        '--description',
        type=str,
        help='brief description of the run and its differences')
    parser.add_argument(
        '--form',
        type=str,
        help='input form (see data.py for more information)',
        default=config.INPUT_FORM,
        )
    parser.add_argument(
        '--label',
        type=str,
        help='label form (see data.py for more information)',
        default="outcome",
        )
    parser.add_argument(
        '--split',
        type=str,
        help='UUID for split',
        default=None,
        )
    parser.add_argument(
        '--hyperparameters',
        type=str,
        help='hyperparameters file',
        required=True,
        )
    parser.add_argument(
        '--trials',
        type=int,
        default=config.TRIALS,
        help='how many times to run')
    FLAGS, unparsed = parser.parse_known_args()
    with open(FLAGS.hyperparameters) as f:
        parameters = json.load(f)
        parameters = explode_parameters(parameters)
    model = models[FLAGS.model]
    split = FLAGS.split
    if split is None:
        split = uuid4()
    else:
        split = UUID(split)
    training, validation, test = data(split, input_form=FLAGS.form, label_form=FLAGS.label)
    for _ in range(FLAGS.trials):
        for hyperparameters in parameters:
            run(model, FLAGS.description, FLAGS.form, FLAGS.label, split, loaded_data=(training, validation, test), hyperparameters=hyperparameters)
            K.clear_session()
