import h5py
from phc_dataset import PhC2DBandgap

train_dataset = PhC2DBandgap(train = True)
test_dataset = PhC2DBandgap(train = False)
print('Loaded Files')
train_inputs = train_dataset[:][0]
train_targets = train_dataset[:][1]
test_inputs = test_dataset[:][0]
test_targets = test_dataset[:][1]


file = h5py.File('bandgap_values.h5', 'w')
file.create_dataset('train_inputs', data=train_inputs)
file.create_dataset('train_targets', data=train_targets)
file.create_dataset('test_inputs', data=test_inputs)
file.create_dataset('test_targets', data=test_targets)
file.close()
print('Created Dataset')