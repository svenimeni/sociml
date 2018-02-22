import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

TRAIN_LIGA = "sampleData/Football-data-2015-1.csv";
TEST_LIGA = "sampleData/Football-data-2016-1.csv";


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['0', '1', '2']

def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

def assignWon(a,b):
    if a>b:
        return '1'
    if a==b:
        return '2'
    return '0'

def load_data(y_name='won'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(TRAIN_LIGA)
    
    # Claculate Winner
    # train['won'] = train[['FTHG', 'FTAG']].apply(assignWon)
   
    train = train.iloc[:,2:6]
    train =  calctotalGoals(train)
    
    train['won'] = train['FTHG'].combine(train['FTAG'], func=assignWon)
    print(train.head())
    train_x, train_y = train, train.pop(y_name)
    print(train_x.head())
    print(train_y.head())
    test = pd.read_csv(TEST_LIGA)
   
   #  test.iloc[:,2:7]
    test = test.iloc[:,2:6]
    test =  calctotalGoals(test)
    test['won'] = test['FTHG'].combine(test['FTAG'], func=assignWon)
    print(test.head())
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def load_datafromfile(filename):
    test = pd.read_csv(filename)
 
    test = test.iloc[:,2:6]
    test = test.dropna(axis=0, how='any')
    
    test =  calctotalGoals(test)
    test['FTHG'] = test['FTHG'].astype(int)
    test['FTAG'] = test['FTAG'].astype(int)
   
    test['won'] = test['FTHG'].combine(test['FTAG'], func=assignWon)
    print(test.head())
    test_x, test_y = test, test.pop('won')

    return (test_x, test_y) 

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def calctotalGoals(test):
    teamgoals = dict() 
    
    hgoals = []
    agoals = []
    pairing = []
    pairings = []
    
    for i in range(len(test)):
        pairing = []
        pairing.append(test['HomeTeam'][i])
        pairing.append(test['AwayTeam'][i])
        pairing = sorted(pairing)
        pairings.append(pairing[0] + '-' + pairing[1])
        
        if (test['HomeTeam'][i] in teamgoals):
             hgoals.append(teamgoals[test['HomeTeam'][i]])
             teamgoals[test['HomeTeam'][i]] += test['FTHG'][i]
        else:
             hgoals.append(0)
             teamgoals[test['HomeTeam'][i]] = test['FTHG'][i]
     
        if (test['AwayTeam'][i] in teamgoals):
             agoals.append(teamgoals[test['AwayTeam'][i]])
             teamgoals[test['AwayTeam'][i]] += test['FTAG'][i]
        else:
             agoals.append(0)
             teamgoals[test['AwayTeam'][i]] = test['FTAG'][i]
             
        
     
    test['homegoalstotal']=hgoals
    test['awaygoalstotal']=agoals
    test['pairing']=pairings
    
    
   # print (teamgoals)    
   # print (hgoals)    
    
    return(test)

# The remainder of this file contains a simple example of a csv parser,
#     implemented using a the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
