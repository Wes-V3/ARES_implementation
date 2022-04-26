# 能用，待优化！
# 使用ARES的pdb文件给相应的pdb文件加label！
import atom3d.datasets.datasets as da
import os

#待提升：argparse
input_dir='./example_train' #这是pdb文件的目录，这里的pdb文件中包含了scores信息
output_dir='./labeled_lmdbs-new_rms/train' #输出lmdb文件的目录

def add_label(item):
    label_file = os.path.join(input_dir, item['id'])
    for line in open(label_file): #待提升：rms在TER之后，从后往前遍历更快！
        list = line.split()
        if list != []:
            if list[0] == 'new_rms': #一开始使用的rms,发现结果不完全相同；所以采用new_rms实验
                item['scores'] = float(list[1]) #这里的scores就是rms，为了防止对于ARES本身代码改动太大，暂时不修改
                #如果不转换为float，label = torch.tensor(label)就会报错ValueError: too many dimensions 'str'。
                # print('rms=', list[1])
                break 

    return item

# Load dataset from directory of PDB files
dataset = da.load_dataset(input_dir, 'pdb', transform=add_label)

# Create LMDB dataset from PDB dataset
da.make_lmdb_dataset(dataset, output_dir)