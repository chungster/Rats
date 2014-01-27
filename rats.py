#!/usr/bin/env python

# datasets from NYC Open Data
# Restaurant inspection grade https://nycopendata.socrata.com/Health/Restaurant-Inspection-Results/4vkw-7nck?
# Rat sightings https://data.cityofnewyork.us/Social-Services/Rat-Sightings/3q43-55fe

"""
Rats sightings vs. restaurant inspection grades script.
"""
import collections
import pprint

import pandas
import pandas.io.parsers
import numpy
##import pylab as pl
import matplotlib.pyplot as plt
from sklearn import svm

# from ggplot import *

#from sklearn.svm import SVC
#from sklearn.cross_validation import train_test_split, StratifiedKFold

numpy.set_printoptions(threshold='nan',
                       linewidth=300)



def gather_and_clean_up_data():

    # Read a table of inspection data.
    inspections = pandas.io.parsers.read_csv("Inspections/WebExtract.txt")

    # Remove some columns we don't need.
    ## INSPDATE, GRADEDATE, RECORDDATE
    keep_columns = set(['BORO', 'ZIPCODE', 'CUISINECODE', 'ACTION', 'VIOLCODE', 'SCORE', 'CURRENTGRADE'])
    for column_name in inspections.columns:
        if column_name not in keep_columns:
        inspections.pop(column_name)

    # Load a table of sightings.
    sightings = pandas.io.parsers.read_csv("Sightings/Rat_Sightings.csv")

    # Compute the number of sightings per zipcode.
    # the result is a dataframe with the zipcode as index and the count as value.
    nb_sightings = sightings.groupby('Incident Zip')['Unique Key'].count()
    nb_sightings = pandas.DataFrame(nb_sightings, columns=['NB_SIGHTINGS'])

    # Join the inspections table to the sightings table in order to add the
    # number of sightings for each restaurant's zipcode.
    return pandas.merge(inspections, nb_sightings,
                        left_on='ZIPCODE', right_index=True)


def plot_cuisine_codes(cuisine_code):
    # Count each of the cuisine-code values and draw a bar chart.
    value_counts = collections.defaultdict(int)
    for code in cuisine_code:
        if code < 0:
            continue
        value_counts[code] += 1

    # FIXME: TODO - add labels to the bar chart.
    pprint.pprint(dict(value_counts))
    plt.bar(value_counts.keys(), value_counts.values())
    plt.show()


def plot_scores(scores):
    plt.hist(scores, range=[numpy.min(scores), numpy.max(scores)], bins=100)
    plt.show()


def nominal_to_binary(nominal):
    """Convert nominal data to binary."""

    # Compute all the unique values.
    unique_values = sorted(set(nominal))
    values_index = dict( (value, index)
                         for (index, value) in enumerate(unique_values) )
    nb_unique_values = len(unique_values)

    # Create a binary array of features.
    binary_array = numpy.zeros( (len(nominal), nb_unique_values), dtype=int)
    for row, value in enumerate(nominal):
        index = values_index[value]
        binary_array[row, index] = 1

    return binary_array


def main():
    if 0:
        # Get original data and write it out to a temporary file.
        dataframe = gather_and_clean_up_data()
        dataframe.to_csv(open('rats.csv', 'w'))
    else:
        # Read it from temporary file.
        dataframe = pandas.io.parsers.read_csv(open('rats.csv'))

    ##print dataframe[:100].to_string(); raise SystemExit

    # Use less data for testing this out. We shuffle in order to get a good mix
    # of all the data, which is naturally ordered by zip code in the input file.
    dataframe.reindex(numpy.random.permutation(dataframe.index))
    dataframe = dataframe[:10000]

    nb_samples = dataframe.shape[0]

    # This is an ordinal data column. A higher number of sightings is worse than
    # a low one.
    nb_sightings = dataframe['NB_SIGHTINGS'].values
    nb_sightings = numpy.reshape(nb_sightings, (nb_sightings.shape[0], 1) )

    # Note: cuisine-code, violation-code, and boro are not ordinal data, so we
    # need to convert them to binary features.
    cuisine_code_nominal = dataframe['CUISINECODE'].values.astype(int)
    cuisine_code = nominal_to_binary(cuisine_code_nominal)

    boro_nominal = dataframe['BORO'].values
    boro = nominal_to_binary(boro_nominal)

    violation_code_nominal = dataframe['VIOLCODE'].values
    violation_code = nominal_to_binary(violation_code_nominal)

    scores = dataframe['SCORE'].fillna(0).values

    # Create data matrix.
    X = numpy.hstack( (nb_sightings, cuisine_code, boro, violation_code) )

    # (Alternative way.)
    # nb_features = 2
    # X = numpy.empty( (nb_samples, nb_features), dtype=int)
    # X[:,0] = nb_sightings
    # X[:,1] = cuisine_code
    # X[:,2] = boro
    # X[:,3] = boro

    # Create labels.
    Y = scores
    # plot_scores(scores)

    # Split training and test sets.
    nb_samples_training = int(0.90 * nb_samples)
    training_X = X[:nb_samples_training,:]
    training_Y = Y[:nb_samples_training]
    test_X     = X[nb_samples_training:,:]
    test_Y     = Y[nb_samples_training:]
    # print X.shape, training_X.shape, test_X.shape

    # Train the SVM.
    clf = svm.SVC()
    clf.fit(training_X, training_Y)
    print 'Done training: {}\n'.format(clf)

    # Evaluate the model against the test set.
    predicted_Y = clf.predict(test_X)

    mad_error = numpy.mean(numpy.abs(predicted_Y - test_Y))
    diff_Y = predicted_Y - test_Y
    rms_error = numpy.sqrt(numpy.mean(diff_Y * diff_Y))
    print "Errors:   MAD: {}   RMS: {}".format(mad_error, rms_error)


if __name__ == '__main__':
    main()