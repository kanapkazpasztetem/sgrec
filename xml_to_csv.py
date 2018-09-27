# clear screen
import os
if ( os.name != 'nt' ):
	os.system( "clear" )
else:
	os.system( "cls" )

import xml.etree.ElementTree as ET
import csv

# =============================================
# settings
# =============================================
# filenames without extention!
input_filename = 'data/feature_values_classical'
output_filename = 'data/sgrec'
test_data_percent = 0.1 # this % data will be saved in <outfilename>_test rest will be in <outfilename>_training

# =============================================
# prepare values
# =============================================
output_file_test = output_filename + '_test.csv'
output_file_training = output_filename + '_training.csv'
input_file = input_filename + '.xml'

tree = ET.parse( input_file )
root = tree.getroot()
count = 0
current_class_number = 0
output_files_exist = 0
if os.path.exists( output_file_test ) and os.path.exists( output_file_training ):
	output_files_exist = 1

# =============================================
# split data ratio
# =============================================
test_data_count = test_data_count_sum = int( len( root.findall( 'data_set' ) ) * test_data_percent )
training_data_count = training_data_count_sum = len( root.findall( 'data_set' ) ) - test_data_count
print( "From this file:" )
print( "  Test data count: " + str( test_data_count ) )
print( "  Training data count: " + str( training_data_count ) )

if output_files_exist == 1:
	with open( output_file_test, 'r' ) as f:
		first_line_test = f.readline()
	csv_test_data_count = first_line_test.split(',')[0]
	csv_classes_count = first_line_test.split(',')[2]

	with open( output_file_training, 'r' ) as f:
		first_line_training = f.readline()
	csv_training_data_count = first_line_training.split(',')[0]
	
	current_class_number = int( csv_classes_count ) + 1
	print( "Already existing data:" )
	print( "  Test data count: " + str( csv_test_data_count ) )
	print( "  Training data count: " + str( csv_training_data_count ) )
	test_data_count_sum += int( csv_test_data_count )
	training_data_count_sum += int( csv_training_data_count )

# =============================================
# set up csv header
# =============================================
data_test = []
data_training = []
try:
	with open( output_file_test, 'r' ) as file:
		# read a list of lines into data
		data_test = file.readlines()
	with open( output_file_training, 'r' ) as file:
		# read a list of lines into data
		data_training = file.readlines()
except:
	print( 'file does not exist' )
	data_test = []
	data_training = []

try:
	data_test[0] = ''
	data_training[0] = ''
except:
	data_test.append( '' )
	data_training.append( '' )

# =============================================
# print csv header
# =============================================
test_head = []
training_head = []
test_head.append( test_data_count_sum )
training_head.append( training_data_count_sum )
features_count = 32
test_head.append( features_count )
training_head.append( features_count )
test_head.append( current_class_number )
training_head.append( current_class_number )
# print( test_head )
# print( training_head )
data_test[0] = str( test_head[0] ) + ',' + str( test_head[1] ) + ',' + str( test_head[2] ) + '\n'
data_training[0] = str( training_head[0] ) + ',' + str( training_head[1] ) + ',' + str( training_head[2] ) + '\n'
# and write everything back
with open( output_file_test, 'w' ) as file:
	for line in data_test:
		file.write( line )
with open( output_file_training, 'w' ) as file:
	for line in data_training:
		file.write( line )

# =============================================
# open a files for writing
# =============================================
output_test_file  = open( output_file_test, 'a', newline='' )
output_training_file  = open( output_file_training, 'a', newline='' )
test_file_writer = csv.writer( output_test_file )
training_file_writer = csv.writer( output_training_file )

# =============================================
# parse
# =============================================
for member in root.findall( 'data_set' ):
	data_set = []
	# if count == 0:

	count = count + 1

	# set up csv row
	for feature in member.findall( 'feature' ):
		feature_val = float( feature.find( 'v' ).text )
		data_set.append( feature_val )
	data_set.append( current_class_number )

	#write to proper file
	if count <= test_data_count:
		test_file_writer.writerow( data_set )
	else:
		training_file_writer.writerow( data_set )
output_test_file.close()
output_training_file.close()