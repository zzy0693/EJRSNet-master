import os
import re
import random

devkit_dir = './VOCdevkit/VOC2007'
output_dir = './'

def get_dir(devkit_dir,  type):
    return os.path.join(devkit_dir, type)

def walk_dir(devkit_dir):
    filelist_dir = get_dir(devkit_dir, 'ImageSets/Main')
    annotation_dir = get_dir(devkit_dir, 'Annotations')
    img_dir = get_dir(devkit_dir, 'JPEGImages')
    trainval_list = []
    train_list = []
    val_list = []
    test_list = []

    added = set()

    for _, _, files in os.walk(filelist_dir):
        for fname in files:
            print(fname)
            img_ann_list = []
            if re.match('trainval.txt', fname):
                img_ann_list = trainval_list
            elif re.match('train.txt', fname):
                img_ann_list = train_list
            elif re.match('val.txt', fname):
                img_ann_list = val_list
            elif re.match('test.txt', fname):
                img_ann_list = test_list
            else:
                continue
            fpath = os.path.join(filelist_dir, fname)
            for line in open(fpath):
                name_prefix = line.strip().split()[0]
                print(name_prefix)

                added.add(name_prefix)
                #ann_path = os.path.join(annotation_dir, name_prefix + '.xml')
                ann_path = annotation_dir + '/' + name_prefix + '.xml'
                print(ann_path)
                #img_path = os.path.join(img_dir, name_prefix + '.jpg')
                img_path = img_dir + '/' + name_prefix + '.jpg'
                assert os.path.isfile(ann_path), 'file %s not found.' % ann_path
                assert os.path.isfile(img_path), 'file %s not found.' % img_path
                img_ann_list.append((img_path, ann_path))
            print(img_ann_list)

    return trainval_list, train_list, val_list, test_list


def prepare_filelist(devkit_dir, output_dir):
    trainval_list = []
    train_list = []
    val_list = []
    test_list = []

    trainval, train, val, test = walk_dir(devkit_dir)

    trainval_list.extend(trainval)
    train_list.extend(train)
    val_list.extend(val)
    test_list.extend(test)
    #print(trainval)
    with open(os.path.join(output_dir, 'trainval.txt'), 'w') as ftrainval:
        for item in trainval_list:
            ftrainval.write(item[0] + ' ' + item[1] + '\n')

    with open(os.path.join(output_dir, 'train.txt'), 'w') as ftrain:
        for item in train_list:
            ftrain.write(item[0] + ' ' + item[1] + '\n')

    with open(os.path.join(output_dir, 'val.txt'), 'w') as fval:
        for item in val_list:
            fval.write(item[0] + ' ' + item[1] + '\n')

    with open(os.path.join(output_dir, 'test.txt'), 'w') as ftest:
        for item in test_list:
            ftest.write(item[0] + ' ' + item[1] + '\n')


if __name__ == '__main__':
    prepare_filelist(devkit_dir, output_dir)


