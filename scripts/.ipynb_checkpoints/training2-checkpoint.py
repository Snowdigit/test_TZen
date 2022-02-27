import tensorflow as tf
from keras.utils import np_utils
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

import imblearn
from imblearn.over_sampling import SMOTE, RandomOverSampler



# La methode de rééquilibrage

sm = RandomOverSampler(random_state=1, sampling_strategy='not majority')
data, y = sm.fit_resample(X,y)



#le meilleur modèle potentiel après rééquibrage

def deep_model():
    """
    Cette fonction représente l'architecture de notre modèle de deep learning.
    En entrée nous avons le nombre de variables explicatives (18) et en sortie le nombre de classe (1436).
    Pour réussir cette classification multi-classes, nous avons utilisé la fonction d'activation softmax en sortie.
    """
    model = Sequential()
    model.add(Dense(4000, input_dim = 18, activation = 'relu')) 
    model.add(Dense(3000, activation = 'relu'))
    model.add(Dense(2000, activation = 'relu'))
    model.add(Dense(1436, activation = 'softmax')) 
    model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
    model.summary()
    return model


#l'entrainement


model = deep_model()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.001)
file_path = '../models/best_model.h5'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
callbacks = [reduce_lr, model_checkpoint]

history = model.fit(xtrain, ytrain, epochs=100, batch_size=32, validation_split=0.1, callbacks=callbacks)



