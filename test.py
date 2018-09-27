import sys
import tensorflow as tf
import os
import shutil

fileprefix = 'data/sgrec'
input_training_file = fileprefix + '_training.csv'
input_test_file = fileprefix + '_test.csv'
model_directory = 'tmp/'

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
  features_labels.extend( ["Feature_" + str( x + 1 ) ] )

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

def calc(batch_size, training_steps, nn_layers_count, nn_layers, learning_rate):
    nn_layerss = []
    for i in range(1, nn_layers_count):
        nn_layerss.append(nn_layers)
    tf.logging.set_verbosity( tf.logging.ERROR )
    # ================
    # Build NN
    # @see https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier#class_dnnclassifier
    # ================
    feature_columns = [tf.feature_column.numeric_column( label, shape = 1 ) for label in features_labels]
    nn_classifier = tf.estimator.DNNClassifier( feature_columns = feature_columns, hidden_units = nn_layerss, n_classes = genres, model_dir = model_directory )
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
    #print('\n-----------==================================-----------')
    #print( '                 Accuracy: ' + str( scores['accuracy'] ) )
    #print('-----------==================================-----------\n')
    print(str(batch_size) + ';' + str(training_steps) + ';' + str(nn_layers) + ';' + str(nn_layers_count) + ';' + str(scores['accuracy']))

    # ================
    # Run tensorboard
    # ================
    # os.system( "python -m tensorboard.main --logdir=" + model_directory )
    #print( "Open new console and paste there this command:\n" )
    #print( "python -m tensorboard.main --logdir=" + model_directory + "\n\n========================")

    #if __name__ == '__main__':
    #tf.app.run()

batch_size_variants = [2,4,8,16,32,64,128,256,512,1024]
training_steps_variants = [100,200,500,1000,2000,5000]
nn_layers_count_variants = [1, 2, 3, 4, 5, 6]
nn_layers_variants = [10, 20, 30, 40, 50]
learning_rate_variants = [0]
for batch_size in batch_size_variants:
    for training_steps in training_steps_variants:
        for nn_layers_count in nn_layers_count_variants:
            for nn_layers in nn_layers_variants:
                for learning_rate in learning_rate_variants:
                    shutil.rmtree(model_directory, ignore_errors=True)
                    #import os
                    #os.system("pause")
                    calc(batch_size, training_steps, nn_layers_count, nn_layers, learning_rate)