import os, numpy as np, pandas as pd, tensorflow as tf, tensorflowjs as tfjs
from sklearn.model_selection import train_test_split

np.random.seed(42); tf.random.set_seed(42)

# 1) Load data
df = pd.read_csv("training.csv")
key = [c for c in df.columns if c != "Image"]
df = df[df.Image.notna() & df.Image.str.len().gt(0)].reset_index(drop=True)

X = np.stack([np.fromstring(s, sep=" ", dtype=np.float32).reshape(96,96,1)/255. for s in df.Image], 0)
Yraw = df[key].values.astype(np.float32)
M = (~np.isnan(Yraw)).astype(np.float32)
Y = np.nan_to_num(Yraw, nan=0.0)
Yp = np.concatenate([Y, M], 1)

Xtr, Xv, Ytr, Yv = train_test_split(X, Yp, test_size=0.15, random_state=42)

# 2) Small CNN
inp = tf.keras.Input((96,96,1))
x = tf.keras.layers.Conv2D(32,3,activation="relu",padding="same")(inp)
x = tf.keras.layers.Conv2D(32,3,activation="relu",padding="same")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(64,3,activation="relu",padding="same")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128,activation="relu")(x)
out = tf.keras.layers.Dense(30)(x)
model = tf.keras.Model(inp,out)

@tf.function
def masked_mse(y_true_pack, y_pred):
    y_true = y_true_pack[:,:30]; mask = y_true_pack[:,30:]
    se = tf.square(y_true - y_pred) * mask
    denom = tf.reduce_sum(mask, axis=1) + 1e-8
    return tf.reduce_mean(tf.reduce_sum(se, axis=1) / denom)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=masked_mse)

# 3) Train (set epochs=3 если очень спешишь)
model.fit(Xtr, Ytr, validation_data=(Xv, Yv), batch_size=64, epochs=10, verbose=1)

# 4) Export Keras + TFJS
os.makedirs("model", exist_ok=True)
model.save("model/fkd_cnn.keras")
tfjs.converters.save_keras_model(model, "model")  # creates model/model.json + shards
print("Saved TFJS to ./model")
