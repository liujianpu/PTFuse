import os


def accuracy(y, t):
    pred = y.data.max(1, keepdim=True)[1]
    acc = pred.eq(t.data.view_as(pred)).cpu().sum()
    return float(acc)


def class_num(real):
    assert len(set(real.tolist())) > 1
    if len(set(real.tolist())) == 2:
        return True
    else:
        return False

def class_dict(args, isindata):
    if args.cls_dataset == "mnist":
        classes = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
        '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
    }
    elif args.cls_dataset == "cifar10":
        classes = {
        'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
        'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
    }
        
    else:
        classes = {}
        test_folder_names = os.listdir(args.cls_dataroot+'/' + args.cls_dataset + '/' + 'test/')
        train_folder_names = os.listdir(args.cls_dataroot+'/' + args.cls_dataset + '/' + 'train/')
        test_folder_names.sort()
        train_folder_names.sort()
        assert len(test_folder_names) == len(train_folder_names),  '''your train folder contain more or less categories than test. '''
        for i in range(len(test_folder_names)):
            classes[test_folder_names[i]] = i
    if isindata:
        return classes
    else:
        classes_reverse = {value:key for key, value in classes.items()}
        return classes_reverse

