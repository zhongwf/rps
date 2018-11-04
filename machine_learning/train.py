#训练

#增加训练的函数，增加训练的report
from keras.callbacks import Callback

class MoreInfoCallback(Callback):
    def __init__(self):
        super(MoreInfoCallback, self).__init__()
    
    def on_epoch_end(self, epoch, logs=None):
        print("MoreInfoCallback work, print value set confusion matrix")
#loadModelAndPredictAndPringConfusionMatrix()

moreInfoCallback = MoreInfoCallback()
print("build moreInfoCallback finish")


#%config ZMQInteractiveShell.ast_node_interactivity='all'
import os
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.mobilenet import MobileNet2
from keras.applications.mobilenetv2 import MobileNetV2
import keras
learning_rate = 0.0001
img_width = 128
img_height = 128
allTargetTypes = ['rock', 'paper', 'scissors', 'room']

nbr_epochs = 25
batch_size = 32

nbr_train_samples = 0
nbr_validation_samples = 0

all_kind = len(allTargetTypes) #几分类

#train_data_dir = '/Users/sharp/dev/project/july/course/my-app/Rock-paper-scissors/my/data_simple/train_split'
#val_data_dir = '/Users/sharp/dev/project/july/course/my-app/Rock-paper-scissors/my/data_simple/val_split'

project_root_dir = os.getcwd()
activeFolder = 'data'
#activeFolder = 'data'
#activeFolder = 'data'

train_data_dir = os.path.join(project_root_dir, activeFolder + '/train_split')
val_data_dir = os.path.join(project_root_dir, activeFolder + '/val_split')

for targetType in allTargetTypes:
    total_train_images = os.listdir(os.path.join(train_data_dir, targetType))
    for img in total_train_images:
        nbr_train_samples += 1
    total_val_images = os.listdir(os.path.join(val_data_dir, targetType))
    for img in total_val_images:
        nbr_validation_samples += 1

print('Loading Model From file ...')
MobileNet_notop = MobileNetV2(include_top=False,input_tensor=None, input_shape=(128, 128, 3))


print('Adding Average Pooling Layer and Softmax Output Layer ...')
output = MobileNet_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)

#output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = AveragePooling2D((4, 4), strides=(4, 4), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(all_kind, activation='softmax', name='predictions')(output)

#建立fine tune模型
InceptionV3_model = Model(MobileNet_notop.input, output)
for layer in InceptionV3_model.layers[-7:]:
    layer.trainable = False
optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy', keras.metrics.categorical_accuracy])

#自动保存文件
best_model_file = "./weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

# # this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
                                   rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   rotation_range=10.,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True)
# # only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

# train_datagen = ImageDataGenerator()
# val_datagen = ImageDataGenerator()

for targetType in allTargetTypes:
    if targetType in os.listdir(os.path.join(project_root_dir, activeFolder + '/train_generator')):
        shutil.rmtree(os.path.join(root_val, targetType))

train_generator = train_datagen.flow_from_directory(
                                                    train_data_dir,
                                                    target_size = (img_width, img_height),
                                                    batch_size = batch_size,
                                                    shuffle = True,
                                                    save_to_dir = os.path.join(project_root_dir, activeFolder + '/train_generator'),
                                                    save_prefix = 'aug',
                                                    classes = allTargetTypes,
                                                    class_mode = 'categorical')
for targetType in allTargetTypes:
    if targetType in os.listdir(os.path.join(project_root_dir, activeFolder + '/validation_generator')):
        shutil.rmtree(os.path.join(root_val, targetType))

validation_generator = val_datagen.flow_from_directory(
                                                       val_data_dir,
                                                       target_size=(img_width, img_height),
                                                       batch_size=batch_size,
                                                       shuffle = True,
                                                       save_to_dir = os.path.join(project_root_dir, activeFolder + '/validation_generator') ,
                                                       save_prefix = 'aug',
                                                       classes = allTargetTypes,
                                                       class_mode = 'categorical')

print("nbr_train_samples:", nbr_train_samples)
print("nbr_validation_samples:", nbr_validation_samples)
#steps_per_epoch = nbr_train_samples/batch_size
#validation_steps = nbr_validation_samples / batch_size
steps_per_epoch = nbr_train_samples
validation_steps = nbr_validation_samples
print("steps_per_epoch:", steps_per_epoch)
print("validation_steps:", validation_steps)

history=InceptionV3_model.fit_generator(
                                        train_generator,
                                        #    samples_per_epoch = nbr_train_samples,
                                        #steps_per_epoch=10,
                                        steps_per_epoch=steps_per_epoch,
                                        nb_epoch = nbr_epochs,
                                        validation_data = validation_generator,
                                        #nb_val_samples = nbr_validation_samples,
                                        validation_steps = validation_steps,
                                        verbose = 1,
                                        callbacks = [best_model,moreInfoCallback])


import pickle
with open('train.results.history', 'wb') as f:
    pickle.dump(history.history, f)

