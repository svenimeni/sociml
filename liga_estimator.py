#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import liga_data
import summaryBoard


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])
    print("***************************************")
    print("******            New Run         *****")
        
    
    # Fetch the data
    (train_x, train_y), (test_x, test_y) = liga_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    # for key in train_x.keys():
       # my_feature_columns.append(tf.feature_column.numeric_column(key=key))
       #  print("Key:" + key + "  - Content" )

    vocabulary_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(key="HomeTeam", vocabulary_file="sampleData/teams.txt", vocabulary_size=18)
    my_feature_columns.append(tf.feature_column.indicator_column(vocabulary_feature_column))
    
    vocabulary_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
        key="AwayTeam",
        vocabulary_file="sampleData/teams.txt",
        vocabulary_size=18)
    
    my_feature_columns.append(tf.feature_column.indicator_column(vocabulary_feature_column)) 
   # my_feature_columns.append(tf.feature_column.numeric_column(key='FTHG'))
   # my_feature_columns.append(tf.feature_column.numeric_column(key='FTAG'))
    my_feature_columns.append(tf.feature_column.numeric_column(key='homegoalstotal'))
    my_feature_columns.append(tf.feature_column.numeric_column(key='awaygoalstotal'))
    
    
    vocabulary_feature_column =  tf.feature_column.categorical_column_with_vocabulary_list(key="FTR", vocabulary_list=["H", "A", "D"])
    #my_feature_columns.append(tf.feature_column.indicator_column(vocabulary_feature_column))   
    
    hashed_feature_column =  tf.feature_column.categorical_column_with_hash_bucket("pairing",150)
    
    my_feature_columns.append(tf.feature_column.indicator_column(hashed_feature_column)) 

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3)

    # Train the Model.
    classifier.train(
        input_fn=lambda:liga_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:liga_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy for test_X: {accuracy:0.3f}\n'.format(**eval_result))

 
    test_x, test_y = liga_data.load_datafromfile('sampleData/Football-data-2017-1.csv')
    eval_result = classifier.evaluate(
        input_fn=lambda:liga_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))
    print('\nTest set accuracy for 2017: {accuracy:0.3f}\n'.format(**eval_result))
    
    test_x, test_y = liga_data.load_datafromfile('sampleData/Football-data-2018-1.csv')
    eval_result = classifier.evaluate(
        input_fn=lambda:liga_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))
    print('\nTest set accuracy for 2018: {accuracy:0.3f}\n'.format(**eval_result))
    
   


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

def old ():
     # Generate predictions from the model
    expected = [1,0]
    predict_x = {
        'HomeTeam': ['Bayern Munich','FC Koln'],
        'AwayTeam': ['Wolfsburg','Hamburg'],
        'FTHG': [2,0],
        'FTAG': [1,0],
    }

    predictions = classifier.predict(
        input_fn=lambda:liga_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))
  
    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(liga_data.SPECIES[class_id],
                              100 * probability, expec))
    