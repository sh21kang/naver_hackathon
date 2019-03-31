# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import nsml
import numpy as np
from rmac import *
from nsml import DATASET_PATH
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation,Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from densenet121 import DenseNet
from keras.preprocessing import image
import scipy.io
import utils
from RoiPooling import RoiPooling
from get_regions import rmac_regions, get_size_vgg_feat_map
N=1000
def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')
     
    def infer(queries, db):
        queries = [v.split('/')[-1].split('.')[0] for v in queries]
        db = [v.split('/')[-1].split('.')[0] for v in db]
        queries.sort()
        db.sort()

        queries, query_vecs, references, reference_vecs = get_feature(model, queries, db)

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)

        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)
        indices = np.argsort(sim_matrix, axis=1)
        indices = np.flip(indices, axis=1)
        
        retrieval_results = {}
        

        queries2_vecs=[]
        for (i, query) in enumerate(queries):
            ranked_list = [references[k] for k in indices[i]]
            ranked_list = ranked_list[:N]
            retrieval_results[query] = ranked_list
            # print('retrieval_results[{1}] = {2}'.format(query,ranked_list))
            queries2_vecs.append(reference_vecs[indices[i][10]])
        
        sim_matrix2=np.dot(queries2_vecs, reference_vecs.T)
        indices2 = np.argsort(sim_matrix2, axis=1)
        indices2 = np.flip(indices2, axis=1)
        retrieval_results2 = {}

        
        new_sim_matrix=sim_matrix
        indices1_at_1000=indices[:,:N]
        indices2_at_1000=indices2[:,:N]

        base=sim_matrix[0]-sim_matrix[-1]
        weight=base/len(query_vecs)
        
        for (i,ref) in enumerate(indices1_at_1000): # ref
            # print("######################")
            for (j, ref2) in enumerate(ref[:N]):
                if ref2 in indices2_at_1000[i]:
                    new_sim_matrix[i][ref2]=new_sim_matrix[i][ref2]+weight*(len(query_vecs)-j)
                    #    print('add weight!!!!1.25')

        new_indices=np.argsort(new_sim_matrix, axis=1)
        new_indices=np.flip(new_indices, axis=1)

                        
        for (i, query) in enumerate(queries):
            ranked_list = [references[k] for k in new_indices[i]]
            ranked_list = ranked_list[:N]
            retrieval_results[query] = ranked_list
            
        # return 0
        return list(zip(range(len(retrieval_results)), retrieval_results.items()))
    nsml.bind(save=save, load=load, infer=infer)
    



def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# data preprocess
def get_feature(model, queries, db):
    img_size = (224, 224)
    test_path = DATASET_PATH + '/test/test_data'

    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('rmac').output)
    test_datagen = ImageDataGenerator(rescale=1. / 255, dtype='float32')
    query_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['query'],
        color_mode="rgb",
        batch_size=1,
        class_mode=None,
        shuffle=False
    )
    query_vecs = intermediate_layer_model.predict_generator(query_generator, steps=len(query_generator), verbose=1)

    reference_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['reference'],
        color_mode="rgb",
        batch_size=1,
        class_mode=None,
        shuffle=False
    )
    reference_vecs = intermediate_layer_model.predict_generator(reference_generator, steps=len(reference_generator),
                                                                verbose=1)

    return queries, query_vecs, db, reference_vecs


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=5)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--num_classes', type=int, default=1383)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes
    input_shape = (224, 224, 3)  # input image shape

    """ Model """
    model=DenseNet(reduction=0.5, classes=num_classes)
    for layer in model.layers[:-3]:
        layer.trainable = False
    bind_model(model)
    # nsml.load(168,session='kh_square/ir_ph2/59')
    nsml.load(78,session='kh_square/ir_ph2/221')
    print('''nsml.load(168,session='kh_square/ir_ph2/59')''')
    file = utils.DATA_DIR + 'sample.jpg'
    img = image.load_img(file)
    img = img.resize((224,224))
    x = image.img_to_array(img)
    print("x=image.img_to_array shape: ",x.shape)
    x = np.expand_dims(x, axis=0)
    print("x=np.expand_dims(x, axis=0): ",x.shape)
    x = utils.preprocess_image(x)
    print("x=utils.preprocess_image(x) ",x.shape)
    # Load RMAC model
    Wmap, Hmap = get_size_vgg_feat_map(x.shape[1], x.shape[2])
    print("get_size_vgg_feat_map(x.shape[3], x.shape[2])",x.shape[1],",",x.shape[2])
    regions = rmac_regions(Wmap, Hmap, 3)
    print("rmac regions\n",regions,"\n rmac-length: ",len(regions))
    in_roi=np.expand_dims(regions, axis=0)
    model = rmac((x.shape[3], x.shape[2], x.shape[1]), len(regions),model,in_roi)
    print('Extracting RMAC from image...')
    print("np.expand_dims(regions, axis=0)",np.expand_dims(regions, axis=0))
    RMAC = model.predict(x)
    print('RMAC size: %s' % RMAC.shape[1])
    print('Done!')
    print('DenseNet121 rmac practice haein epoch',nb_epoch)
    bind_model(model)
    model.summary()
    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Initiate RMSprop optimizer """
        opt = keras.optimizers.rmsprop(lr=0.00045, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        print('dataset path', DATASET_PATH)

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        #local_dir = './train/train_data'
        #directory=DATASET_PATH + '/train/train_data',
        train_generator = train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            target_size=input_shape[:2],
            color_mode="rgb",
            #batch_size=batch_size,
            batch_size=1,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )

        """ Callback """
        monitor = 'acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        #t0 = time.time()
        for epoch in range(nb_epoch):
           #t1 = time.time()
           #res = model.fit_generator(generator=train_generator,
           #                          steps_per_epoch=STEP_SIZE_TRAIN,
           #                          initial_epoch=epoch,
           #                          epochs=epoch + 1,
           #                          callbacks=[reduce_lr],
           #                          verbose=1,
           #                          shuffle=True)
           #t2 = time.time()
           #print(res.history)
           #print('Training time for one epoch : %.1f' % ((t2 - t1)))
           #train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
           #nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)
           nsml.save(epoch)
        #print('Total training time : %.1f' % (time.time() - t0))
        print("done")
