import csv
import os
import argparse
import glob
import nrrd
import re
import numpy as np
import pandas
from sklearn import preprocessing as sklearn_preprocessing

from config import config
from filenames import ACCEPTED_FILENAMES
from segmentation import calculate_volume

clinical_feature_functions = {
    "outcome": lambda f: "malignant" if f["Malignancy-binary"] == "1" else "benign",
    "age": lambda f: int(f["Age"]),
    "sex": lambda f: 1 if f["Sex"] == "M" else 0,
    "sort": lambda f: f["sort"],
    "pathology": lambda f: f["pathology"],
    "usable": lambda f: f["usable"],
}

def clinical_features(feat, filename):
    patient = filename_features(filename)["accession"]
    clinical = feat.get(patient, None)
    if clinical is None:
        print("missing from clinical feature sheet: {}".format(patient))
        return {}
    return { k: f(clinical) for k, f in clinical_feature_functions.items() }

image_feature_functions = {
    "volume": calculate_volume,
}

def image_features(filename):
    array, metadata = nrrd.read(filename)
    return { k: f(array, metadata, filename) for k, f in image_feature_functions.items() }

def all_nrrd(folder="."):
    return glob.glob("{}/**/*.nrrd".format(folder), recursive=True)

def all_features(files=["./features.csv"], id_name="PatientID"):
    by_file = list()
    for filename in files:
        with open(filename) as f:
            l = [ {k: v for k, v in row.items() } for row in csv.DictReader(f, skipinitialspace=True )]
            by_accession = { d[id_name]: d for d in l }
            by_file.append(by_accession)
    id_sets = [ set(f.keys()) for f in by_file ]
    union = id_sets[0]
    for ids in id_sets:
        union = union | ids
    combined = dict()
    for i in union:
        c = dict()
        for by_accession in by_file:
            c = {
                **c,
                **by_accession[i],
            }
        combined[i] = c
    return combined

def filename_features(path):
    split_path = path.split(os.sep)
    filename = split_path[-1]
    modality = split_path[-2]
    accession = split_path[-3]
    patient = accession
    return {
        "accession": accession,
        "patient": patient,
        "modality": modality,
        "filename": filename,
        "path": path,
    }

def filter_filenames(df):
    df = df[df.filename.isin(ACCEPTED_FILENAMES)]
    return df

def features(df):
    df = df[df.modality=="T1C"][df.filename=="segMask_tumor.nrrd"][["accession", "patient", "volume", *list(clinical_feature_functions.keys())]]
    df = df.set_index("accession")
    df = df.dropna()
    return df

def preprocessing(df):
    df = df.set_index(["accession", "filename", "modality"])
    df = df.loc[~df.index.duplicated(keep='first')]
    df = df.unstack()
    df = df.unstack()
    return df

def normalize_column(df, column=""):
    min_max_scaler = sklearn_preprocessing.MinMaxScaler()
    x = df[[column]].values.astype(float)
    x_scaled = min_max_scaler.fit_transform(x)
    x_scaled = list(zip(*list(x_scaled)))[0]
    df[column] = pandas.Series(x_scaled, index=df.index)
    return df

def run(folder, features_files, out, save=True, nrrd_pickle="", features_pickle="", to_preprocess_pickle=""):
    nrrds = all_nrrd(folder)
    feat = all_features(features_files)

    # create all features
    nrrd_features = pandas.DataFrame(
        [{
            **filename_features(n),
            **clinical_features(feat, n),
            **image_features(n),
        } for n in nrrds])
    nrrd_features = filter_filenames(nrrd_features)
    nrrd_features = nrrd_features.dropna()

    nrrd_features = normalize_column(nrrd_features, column="age")
    nrrd_features = normalize_column(nrrd_features, column="volume")

    features_to_use = features(nrrd_features)

    to_preprocess = preprocessing(nrrd_features)

    if save:
        nrrd_features.to_csv(os.path.join(out, "nrrd-features.csv"))
        nrrd_features.to_pickle(nrrd_pickle)
        features_to_use.to_csv(os.path.join(out, "training-features.csv"))
        features_to_use.to_pickle(features_pickle)
        to_preprocess.to_csv(os.path.join(out, "preprocess.csv"))
        to_preprocess.to_pickle(to_preprocess_pickle)
    else:
        print(to_preprocess.head())
        print(nrrd_features.head())
        print(features_to_use.head())
    return nrrd_features, features_to_use, to_preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folder',
        type=str,
        default=config.RAW_NRRD_ROOT,
        help='raw folder directory')
    parser.add_argument(
        '--features',
        type=str,
        nargs="+",
        default=config.RAW_FEATURES,
        help='csv files of features')
    parser.add_argument(
        '--out',
        type=str,
        default=config.FEATURES_DIR,
        help='output folder')
    parser.add_argument(
        '--nrrd',
        type=str,
        default=config.NRRD_FEATURES,
        help='nrrd features')
    parser.add_argument(
        '--pickle',
        type=str,
        default=config.FEATURES,
        help='features pickle')
    parser.add_argument(
        '--preprocess',
        type=str,
        default=config.PREPROCESS,
        help='preprocess pickle')
    parser.add_argument(
        '--temp',
        action='store_true',
        help='do not save')
    FLAGS, unparsed = parser.parse_known_args()
    run(FLAGS.folder, FLAGS.features, FLAGS.out, not FLAGS.temp, FLAGS.nrrd, FLAGS.pickle, FLAGS.preprocess)
