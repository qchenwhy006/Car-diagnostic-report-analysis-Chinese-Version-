import tensorflow as tf
import tensorflow.keras.backend as K


def micro_f1(y_true,y_pred):
    TP=K.cast(K.sum(K.round(K.clip(y_true*y_pred,0,1)),axis=0),tf.float32)
    predicted_positives=K.cast(K.sum(K.round(K.clip(y_pred,0,1)),axis=0),tf.float32)
    real_positives=K.cast(K.sum(K.round(K.clip(y_true,0,1)),axis=0),tf.float32)

    precision=K.sum(TP)/(K.sum(predicted_positives) + K.epsilon())
    recall=K.sum(TP)/(K.sum(real_positives) + K.epsilon())
    micro_f1 = 2 * precision * recall / (precision + recall + K.epsilon() )

    return micro_f1


def macro_f1(y_true,y_pred):
    TP =K.cast(K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0),tf.float32)
    predicted_positives =K.cast(K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0),tf.float32)
    real_positives = K.cast(K.sum(K.round(K.clip(y_true, 0, 1)), axis=0),tf.float32)

    precision = TP/(predicted_positives+K.epsilon())
    recall = TP/(real_positives+K.epsilon())
    macro_f1 = K.mean(2 * precision * recall/(precision + recall+K.epsilon()))


    return macro_f1