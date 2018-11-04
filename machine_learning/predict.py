#验证

#验证

#my
from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


img_width = 128
img_height = 128
batch_size = 32


FishNames = ['rock', 'paper', 'scissors', 'room']



#root_path = '/Users/sharp/dev/project/july/course/my-app/Rock-paper-scissors/my/'
root_path = os.getcwd()
#nbr_test_samples = 40
#weights_path = os.path.join(root_path, 'weights.h5')
#test_data_dir = os.path.join(root_path, 'data_simple/val_split')




nbr_test_samples = 0
allTargetTypes = ['rock', 'paper', 'scissors', 'room']

weights_path = os.path.join(root_path, 'weights.20181022.h5')
test_data_dir = os.path.join(root_path, 'data/train_split')

all_kind = len(allTargetTypes) #几分类


project_root_dir = os.getcwd()
activeFolder = 'data'

val_data_dir = os.path.join(project_root_dir, activeFolder + '/val_split')

for targetType in allTargetTypes:
    total_test_images = os.listdir(os.path.join(test_data_dir, targetType))
    for img in total_test_images:
        nbr_test_samples += 1



print(1,test_data_dir)
# test data generator for prediction
test_datagen = ImageDataGenerator()
print(2,test_datagen)
print(test_data_dir)
test_generator = test_datagen.flow_from_directory(
                                                  test_data_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  shuffle = False, # Important !!!
                                                  classes = FishNames,
                                                  class_mode = 'categorical'
                                                  )

test_image_list = test_generator.filenames
print("ha",test_image_list)

print('Loading model and weights from training process ...')
InceptionV3_model = load_model(weights_path)
print(InceptionV3_model)
print('Begin to predict for testing data ...')
print(test_generator)
#print(nbr_test_samples)
predictions = InceptionV3_model.predict_generator(test_generator,steps=nbr_test_samples/batch_size)
#, nbr_test_samples

np.savetxt(os.path.join(root_path, 'predictions.txt'), predictions)

print('Begin to write submission file ..')
f_submit = open(os.path.join(root_path, 'submit.csv'), 'w')
f_submit.write('image,rock,paper,scissors,room\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()


print('Submission file successfully generated!')
#predictions
predictions.shape



