from scipy import io
import numpy as np

def OneHot(label,n_classes):
    label=np.array(label).reshape(-1)
    label=np.eye(classes)[label]

    return label

'''If your System is very powerful you can also use the extra.mat file(Extra Data) which can be used for training an validation

I have not created a download function to download and extract the data.This process i have done manually.

Replace train_data,train_labels with extra_data_save,extra_labels_save and use the extra_data and extra_labels and set the
train size accordingly.'''

data1=io.loadmat('train.mat')
data2=io.loadmat('test.mat')
#data3=io.loadmat('extra.mat')

train_data=data1['X']
train_labels=data1['y']
test_data=data2['X']
test_labels=data2['y']
#extra_data=data3['X']
#extra_labels=data3['y']

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
#extra_data=extra_data.astype('float32')

train_data = np.transpose(train_data, (3, 0, 1, 2))
test_data = np.transpose(test_data,(3, 0, 1, 2))

#extra_data=np.transpose(extra_data,(3,0,1,2))


train_labels[train_labels==10]=0

test_labels[test_labels==10]=0

#extra_labels[extra_labels==10]=0

classes=10

train_labels = train_labels[:,0]
test_labels = test_labels[:,0]
#extra_labels=extra_labels[:,0]

train_labels=OneHot(train_labels,classes)
test_labels=OneHot(test_labels,classes)
#extra_labels=OneHot(extra_labels,classes)

print( 'Train data:', train_data.shape,', Train labels:', train_labels.shape )
print( 'Test data:', test_data.shape,', Test labels:', test_labels.shape )

from sklearn.model_selection import train_test_split
import pickle

train_data,validation_train,train_labels,validation_label_final=train_test_split(train_data,train_labels,train_size=70000,random_state=106)
print ('Train data:', train_data.shape,', Train labels:', train_labels.shape)
print ('Validation data:', validation_train.shape,', Validation labels:', validation_label_final.shape)
pickle_file='extra.pickle'

dict_to_pickle={
    'train_dataset':
        {
            'X':train_data,
            'y':train_labels
        },
'test_dataset':
        {
            'X':test_data,
            'y':test_labels
        },

'valid_dataset':
    {
            'X':validation_train,
            'y':validation_label_final
    },

}

with open(pickle_file,'wb') as f:
    print("saving Pickle.W8")
    pickle.dump(dict_to_pickle,f,protocol=pickle.HIGHEST_PROTOCOL)

print("Saved")
