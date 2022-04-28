import os
import pandas as pd
import atom3d.datasets.datasets as da

PATH_TO_LABELS_DIR='./train_csv'
PATH_TO_INPUT_DIR='./example_train'
PATH_TO_LMDB_OUTPUT='./train_csv'

def add_label(item):
    # Remove the file ending ".pdb" from the ID
	name = item['id'][0:4]
	# print(item['id'])
	# print(item['id'][0:4]) #157d
	# print(item['id'][1:4]) #57d
	# print(item['id'][0:3]) #157
	# print(item['id'][1:3]) #57
	if name == '1mhk':
		name = '1mhk_noC'
	elif name == '1l2x':
		name = 'pdb1l2x_A' # 选A还是B
	elif name == '1q9a':
		name = '1q9a_A' # 选A还是B
	
    # Get label data
	label_file = os.path.join(PATH_TO_LABELS_DIR, name+'_rmsds.csv')
    # Add label data in form of a data frame
	rmsds_csv = pd.read_csv(label_file, index_col=0) # 索引无用，还会使按名称筛选变难
	# print(rmsds_csv)
	# print(rmsds_csv.loc[item['id']])

	item['scores'] = float(rmsds_csv.loc[item['id']]) # pandas的dataframe默认选列！用.loc选行
	# print('label = ', item['label'])
	return item

# Load dataset from directory of PDB files
dataset = da.load_dataset(PATH_TO_INPUT_DIR, 'pdb', transform=add_label)

# Create LMDB dataset from PDB dataset
da.make_lmdb_dataset(dataset, PATH_TO_LMDB_OUTPUT)
