# from pprint import pprint # for debugging
import tensorflow as tf

#clear screen
import os
if ( os.name != 'nt' ):
    os.system( "clear" )
else:
    os.system( "cls" )
print('===========================================================')
print('===========================================================')

# =============================================
# Settings
# =============================================

fileprefix = 'data/sgrec'
input_training_file = fileprefix + '_training.csv'
input_test_file = fileprefix + '_test.csv'
model_directory = 'tmp/'
random_seed = 1234

batch_size = 32
training_steps = 1000
nn_layers = [20, 20, 20] # [10, 10, 10] ==> 3 layers with 10 units in each
learning_rate = 0.01

# =============================================
# Set up variables
# =============================================
tf.set_random_seed( random_seed ) # Make results reproducible

with open( input_training_file, 'r' ) as f:
    first_line_test = f.readline()
test_data_count = int( first_line_test.split(',')[0] )
features_count = int( first_line_test.split(',')[1] )
genres = int( first_line_test.split(',')[2] ) + 1

with open( input_test_file, 'r' ) as f:
    first_line_training = f.readline()
training_data_count = int( first_line_training.split(',')[0] )

features_labels = [] # placeholder - we create them in a loop anyway
for x in range( features_count ):
  features_labels.extend( [ "Feature_" + str( x + 1 ) ] )

# =============================================
# Creates an input_fn required by Estimator train/evaluate.
# =============================================
def input_fn( _file_name, _data_count, _batch_size, _is_training ):

  # =============================================
  # Takes the string input tensor and returns tuple of (features, labels).
  # Last dim is the label.  
  # =============================================
  def _parse_csv( _rows_string_tensor ):

    _features_count = features_count
    csv_total_cols = _features_count + 1 # because genre is last col
    columns = tf.decode_csv( _rows_string_tensor, record_defaults=[[]] * csv_total_cols )
    features = dict( zip( features_labels, columns[:_features_count] ) )
    labels = tf.cast( columns[_features_count], tf.int32 )
    return features, labels

  # =============================================
  # The input_fn.
  # =============================================
  def _input_fn():
    dataset = tf.data.TextLineDataset( [_file_name] )
    # Skip the first line - contains config not data
    dataset = dataset.skip( 1 )
    dataset = dataset.map( _parse_csv )

    if _is_training:
      # Shuffle data for training because it is sorted by genre in the files
      dataset = dataset.shuffle( _data_count )
      dataset = dataset.repeat()

    dataset = dataset.batch( _batch_size )
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

  return _input_fn


def main( unused ):
  # Logging
  tf.logging.set_verbosity( tf.logging.INFO )

  # ================
  # Build NN
  # @see https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier#class_dnnclassifier
  # ================
  feature_columns = [tf.feature_column.numeric_column( label, shape = 1 ) for label in features_labels]
  nn_classifier = tf.estimator.DNNClassifier( feature_columns = feature_columns, hidden_units = nn_layers, n_classes = genres, model_dir = model_directory, optimizer = tf.train.GradientDescentOptimizer( learning_rate = learning_rate ), config = tf.estimator.RunConfig().replace( save_summary_steps = 10 ) )
 
  # ================
  # Train
  # ================
  train_input_fn = input_fn( input_training_file, training_data_count, _batch_size = batch_size, _is_training = True )
  nn_classifier.train( input_fn = train_input_fn, steps = training_steps )

  # ================
  # Eval
  # ================
  test_input_fn = input_fn( input_test_file, test_data_count, _batch_size = batch_size, _is_training = False )
  scores = nn_classifier.evaluate( input_fn = test_input_fn )

  print('\n-----------==================================-----------')
  print( '                 Accuracy: ' + str( scores['accuracy'] ) )
  print('-----------==================================-----------\n')

  # ================
  # Run tensorboard
  # ================
  # os.system( "python -m tensorboard.main --logdir=" + model_directory )
  print( "Open new console and paste there this command:\n" )
  print( "python -m tensorboard.main --logdir=" + model_directory + "\n\n========================")

if __name__ == '__main__':
  tf.app.run()