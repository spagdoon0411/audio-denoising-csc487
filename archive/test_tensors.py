import tensorflow as tf 
import numpy as np

def normalize_tensor(tens):
    tens1 = tf.cast(tens, dtype=tf.float32)
    min = tf.math.reduce_min(tens1)
    max = tf.math.reduce_max(tens1)
    shape = tf.shape(tens1)
    ones = tf.ones(shape)

    if(min == max):
        return tf.zeros(shape)

    all_min = tf.math.scalar_mul(min, ones)
    min_zero_tens = tf.math.subtract(tens1, all_min)
    zero_one_tens = tf.math.scalar_mul(
        tf.math.reciprocal(
            tf.math.subtract(max, min)
        ), 
            min_zero_tens
    )

    all_zero_point_five = tf.math.scalar_mul(tf.constant(0.5), ones)
    centered_tens = tf.math.subtract(zero_one_tens, all_zero_point_five)
    normalized_tens = tf.math.scalar_mul(tf.constant(2.0), centered_tens)


    return normalized_tens

t1 = tf.convert_to_tensor(np.array([[-1.0, 3], [-4, 1]]), dtype=tf.float32)
t2 = tf.convert_to_tensor(np.random.rand(4, 12))

tf.print(normalize_tensor(t1))
tf.print(normalize_tensor(t2))

