import cv2

from tensorflow.keras.datasets import cifar100

fine_label = ['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle',
'bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle',
'caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab',
'crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl',
'hamster','house','kangaroo','computer_keyboard','lamp','lawn_mower','leopard',
'lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse',
'mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck',
'pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon',
'ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper',
'snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table',
'tank','telephone','television','tiger','tractor','train','trout','tulip','turtle',
'wardrobe','whale','willow_tree','wolf','woman','worm',
]

coarse_label = [
    'aquatic mammals',
    'fish',
    'flowers',
    'food containers',
    'fruit and vegetables',
    'household electrical device',
    'household furniture',
    'insects',
    'large carnivores',
    'large man-made outdoor things',
    'large natural outdoor scenes',
    'large omnivores and herbivores',
    'medium-sized mammals',
    'non-insect invertebrates',
    'people',
    'reptiles',
    'small mammals',
    'trees',
    'vehicles 1',
    'vehicles 2',
]

# maps fine label to coarse label
mapping = {
'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
'household electrical device': ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],
'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
'people': ['baby', 'boy', 'girl', 'man', 'woman'],
'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}

def resize_cifar100(fine_labelled=True, size=(256,256)):
    label_mode = 'fine' if fine_labelled else 'coarse'
    label = fine_label if fine_labelled else coarse_label
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode=label_mode)
    for i, (data, label_idx) in enumerate(zip(x_train, y_train)):
        resized_image = cv2.resize(data, size)
        cv2.imwrite("../dataset/cifar100/{}/train/{}.{}.jpg".format(label_mode,label[label_idx[0]], i), resized_image)

    for i, (data, label_idx) in enumerate(zip(x_test, y_test)):
        resized_image = cv2.resize(data, size)
        cv2.imwrite("../dataset/cifar100/{}/test/{}.{}.jpg".format(label_mode,label[label_idx[0]], i), resized_image)




resize_cifar100(fine_labelled=False)
