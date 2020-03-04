from config import config
from data import data, xdata, load_from_features
from keras.models import load_model
from keras import backend as K
import math
from uuid import uuid4, UUID

from db import Result, XResult

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import pandas

def load(result):
    path = "{}/models/{}-{}.h5".format(config.OUTPUT, result.uuid, result.model)
    return load_model(path)

def xload(result):
    path = "{}/models/{}-{}.h5".format(config.OUTPUT, result.run_id, result.model)
    return load_model(path)

def stacked_single(uuids=[], pickle="", source=""):
    results = [ Result.query.filter(Result.uuid == uuid).first() for uuid in uuids ]
    modalities = dict()
    for result in results:
        if result.input_form in modalities:
            continue
        features = pandas.read_pickle(pickle)
        dataset = load_from_features(
            features,
            label_form=result.label_form,
            input_form=result.input_form,
            shuffle=False,
            augment=False,
            source=source,
        )
        modalities[result.input_form] = dataset
    # generate labels
    labels = list()
    first = list(modalities.values())[0]
    for _ in range(math.ceil(len(first)/config.BATCH_SIZE)):
        labels += first.next()[1].tolist()
    first.reset()
    # generate predictions
    predictions = list()
    for result in results:
        model = load(result)
        dataset = modalities[result.input_form]
        predictions.append(model.predict_generator(dataset, steps=math.ceil(len(dataset)/config.BATCH_SIZE)).flatten())
        dataset.reset()
        K.clear_session()
        del model
    return predictions, labels, modalities

def stacked_data(
        uuids=[],
        epochs=config.EPOCHS,
        ):
    results = [ Result.query.filter(Result.uuid == uuid).first() for uuid in uuids ]
    assert len(results) > 0, "no models found"
    assert len(set([result.split_uuid for result in results])) == 1, "all models must be trained on the same split"
    assert len(set([result.label_form for result in results])) == 1, "all models must be trained on the same label"
    training = dict()
    training_fixed = dict()
    validation = dict()
    test = dict()
    for result in results:
        if result.input_form in training:
            continue
        t, v, te = data(
                seed=UUID(result.split_uuid),
                input_form=result.input_form,
                label_form=result.label_form,
                train_shuffle=True,
                validation_shuffle=False,
                train_augment=True,
                validation_augment=False,
            )
        tf, _, _ = data(
                seed=UUID(result.split_uuid),
                input_form=result.input_form,
                label_form=result.label_form,
                train_shuffle=False,
                validation_shuffle=False,
                train_augment=False,
                validation_augment=False,
            )
        training[result.input_form] = t
        training_fixed[result.input_form] = tf
        validation[result.input_form] = v
        test[result.input_form] = te
    # generate labels
    train_labels = list()
    training_fixed_labels = list()
    validation_labels = list()
    test_labels = list()
    first_training = list(training.values())[0]
    first_training_fixed = list(training_fixed.values())[0]
    first_validation = list(validation.values())[0]
    first_test = list(test.values())[0]
    for _ in range(epochs):
        train_labels += first_training.next()[1].tolist()
    for _ in range(math.ceil(len(first_validation)/config.BATCH_SIZE)):
        validation_labels += first_validation.next()[1].tolist()
    for _ in range(math.ceil(len(first_training_fixed)/config.BATCH_SIZE)):
        training_fixed_labels += first_training_fixed.next()[1].tolist()
    for _ in range(math.ceil(len(first_test)/config.BATCH_SIZE)):
        test_labels += first_test.next()[1].tolist()
    first_training.reset()
    first_training_fixed.reset()
    first_validation.reset()
    first_test.reset()
    # generate predictions
    train_predictions = list()
    train_fixed_predictions = list()
    validation_predictions = list()
    test_predictions = list()
    for result in results:
        model = load(result)
        t = training[result.input_form]
        tf = training_fixed[result.input_form]
        v = validation[result.input_form]
        te = test[result.input_form]
        train_predictions.append(model.predict_generator(t, steps=epochs).flatten())
        train_fixed_predictions.append(model.predict_generator(tf, steps=math.ceil(len(tf)/config.BATCH_SIZE)).flatten())
        validation_predictions.append(model.predict_generator(v, steps=math.ceil(len(v)/config.BATCH_SIZE)).flatten())
        test_predictions.append(model.predict_generator(te, steps=math.ceil(len(te)/config.BATCH_SIZE)).flatten())
        t.reset()
        tf.reset()
        v.reset()
        te.reset()
        K.clear_session()
        del model
    return train_predictions, validation_predictions, test_predictions, train_labels, validation_labels, test_labels, train_fixed_predictions, training_fixed_labels, (training, training_fixed, validation, test)

def xstacked_data(
        uuids=[],
        epochs=config.EPOCHS,
        ):
    results = [ XResult.query.filter(XResult.run_id == uuid).first() for uuid in uuids ]
    assert len(results) > 0, "no models found"
    assert len(set([result.split for result in results])) == 1, "all models must be trained on the same split"
    assert len(set([result.label_form for result in results])) == 1, "all models must be trained on the same label"
    training = dict()
    training_fixed = dict()
    validation = dict()
    test = dict()
    # holdout = dict()

    for result in results:
        if result.input_form in training:
            continue

        split = UUID(result.split)

        f = pandas.read_pickle(config.FEATURES)
        y = f[result.label_form].values

        # k_fold_train, holdout_test, y_train, y_test = train_test_split(f, y, test_size=config.MAIN_TEST_HOLDOUT, stratify=y, random_state=int(split) % 2 ** 32)

        # set up the k-fold process
        skf = StratifiedKFold(n_splits=config.NUMBER_OF_FOLDS, random_state=int(split) % 2 ** 32)

        # get the folds and loop over each fold
        fold_number = 0
        for train_index, test_index in skf.split(f, y):
            fold_number += 1

            if fold_number != result.fold:
                continue

            # get the training and testing set for the fold
            X_train, testing = f.iloc[train_index], f.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # split the training into training and validation
            train, valid, result_train, result_test = train_test_split(X_train, y_train,
                                                                               test_size=config.SPLIT_TRAINING_INTO_VALIDATION,
                                                                               stratify=y_train,
                                                                               random_state=int(split) % 2 ** 32)

            # train_generator, validation_generator, test_generator, holdout_test_generator
            train_generator, validation_generator, test_generator = xdata(fold_number, train, valid,
                                                                                            testing,  # holdout_test,
                                                                                            split,
                                                                                            input_form=result.input_form,
                                                                                            label_form=result.label_form,
                                                                                            train_shuffle=True,
                                                                                            validation_shuffle=False,
                                                                                            train_augment=True,
                                                                                            validation_augment=False)

            # train_generator_f, validation_generator_f, test_generator_f, holdout_test_generator_f
            train_generator_f, validation_generator_f, test_generator_f = xdata(fold_number, train, valid,
                                                                                                    testing,  # holdout_test,
                                                                                                    split,
                                                                                                    input_form=result.input_form,
                                                                                                    label_form=result.label_form,
                                                                                                    train_shuffle=False,
                                                                                                    validation_shuffle=False,
                                                                                                    train_augment=False,
                                                                                                    validation_augment=False)

            training[result.input_form] = train_generator
            training_fixed[result.input_form] = train_generator_f
            validation[result.input_form] = validation_generator
            test[result.input_form] = test_generator
            # holdout[result.input_form] = holdout_test_generator

    # generate labels
    train_labels = list()
    training_fixed_labels = list()
    validation_labels = list()
    test_labels = list()
    holdout_labels = list()

    first_training = list(training.values())[0]
    first_training_fixed = list(training_fixed.values())[0]
    first_validation = list(validation.values())[0]
    first_test = list(test.values())[0]
    # first_holdout = list(holdout.values())[0]

    for _ in range(epochs):
        train_labels += first_training.next()[1].tolist()

    for _ in range(math.ceil(len(first_validation)/config.BATCH_SIZE)):
        validation_labels += first_validation.next()[1].tolist()

    for _ in range(math.ceil(len(first_training_fixed)/config.BATCH_SIZE)):
        training_fixed_labels += first_training_fixed.next()[1].tolist()

    for _ in range(math.ceil(len(first_test)/config.BATCH_SIZE)):
        test_labels += first_test.next()[1].tolist()

    # for _ in range(math.ceil(len(first_holdout)/config.BATCH_SIZE)):
    #    holdout_labels += first_holdout.next()[1].tolist()

    first_training.reset()
    first_training_fixed.reset()
    first_validation.reset()
    first_test.reset()
    # first_holdout.reset()

    # generate predictions
    train_predictions = list()
    train_fixed_predictions = list()
    validation_predictions = list()
    test_predictions = list()
    # holdout_predictions = list()

    for result in results:
        model = xload(result)

        t = training[result.input_form]
        tf = training_fixed[result.input_form]
        v = validation[result.input_form]
        te = test[result.input_form]
        # h = holdout[result.input_form]

        train_predictions.append(model.predict_generator(t, steps=epochs).flatten())
        train_fixed_predictions.append(model.predict_generator(tf, steps=math.ceil(len(tf)/config.BATCH_SIZE)).flatten())
        validation_predictions.append(model.predict_generator(v, steps=math.ceil(len(v)/config.BATCH_SIZE)).flatten())
        test_predictions.append(model.predict_generator(te, steps=math.ceil(len(te)/config.BATCH_SIZE)).flatten())
        # holdout_predictions.append(model.predict_generator(h, steps=math.ceil(len(h) / config.BATCH_SIZE)).flatten())

        t.reset()
        tf.reset()
        v.reset()
        te.reset()
        # h.reset()
        K.clear_session()
        del model

    return train_predictions, validation_predictions, test_predictions, train_labels, validation_labels, test_labels, train_fixed_predictions, training_fixed_labels  # holdout_predictions, holdout_labels
