# build_fkd_app.py
# Train CNN with proper masked loss (no Keras sample_weight mismatch)
# Export to TFJS and scaffold a static web app.

import os, json, numpy as np, pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_CSV = "training.csv"
EXPORT_DIR = "web"
MODEL_DIR  = os.path.join(EXPORT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------- 1) Load & preprocess --------
df = pd.read_csv(DATA_CSV)
key_cols = [c for c in df.columns if c != "Image"]
assert len(key_cols) == 30, f"Expected 30 keypoint columns, got {len(key_cols)}"

def parse_image_str(s):
    arr = np.fromstring(s, sep=" ", dtype=np.float32) / 255.0
    return arr.reshape(96, 96, 1)

df = df[df["Image"].notna() & df["Image"].str.len().gt(0)].reset_index(drop=True)
X = np.stack([parse_image_str(s) for s in df["Image"].values], axis=0)

Y_raw = df[key_cols].values.astype(np.float32)     # contains NaNs
M     = (~np.isnan(Y_raw)).astype(np.float32)      # mask: 1 if present else 0
Y     = np.nan_to_num(Y_raw, nan=0.0)

# Pack (targets || mask) along last axis → shape (N, 60)
Y_pack = np.concatenate([Y, M], axis=1)

X_train, X_val, Yp_train, Yp_val = train_test_split(
    X, Y_pack, test_size=0.15, random_state=SEED
)

print("Train:", X_train.shape, Yp_train.shape)
print("Val:  ", X_val.shape,   Yp_val.shape)

# -------- 2) Model --------
inputs = tf.keras.Input(shape=(96, 96, 1))
x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPool2D()(x); x = tf.keras.layers.Dropout(0.1)(x)

x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPool2D()(x); x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPool2D()(x); x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
coords = tf.keras.layers.Dense(30, name="coords")(x)

model = tf.keras.Model(inputs, coords)
model.summary()

# -------- 3) Masked loss (split y_true into target & mask) --------
@tf.function
def masked_mse(y_true_pack, y_pred):
    # y_true_pack: [batch, 60] = [targets(30), mask(30)]
    y_true = y_true_pack[:, :30]
    mask   = y_true_pack[:, 30:]
    # squared error per element
    se = tf.square(y_true - y_pred) * mask
    # avoid division by 0
    denom = tf.reduce_sum(mask, axis=1) + 1e-8
    # mean over visible coords, then mean over batch
    per_sample = tf.reduce_sum(se, axis=1) / denom
    return tf.reduce_mean(per_sample)

@tf.function
def masked_mae_metric(y_true_pack, y_pred):
    y_true = y_true_pack[:, :30]
    mask   = y_true_pack[:, 30:]
    ae = tf.abs(y_true - y_pred) * mask
    denom = tf.reduce_sum(mask, axis=1) + 1e-8
    per_sample = tf.reduce_sum(ae, axis=1) / denom
    return tf.reduce_mean(per_sample)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=masked_mse,
    metrics=[masked_mae_metric]
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
]

history = model.fit(
    X_train, Yp_train,
    validation_data=(X_val, Yp_val),
    batch_size=64, epochs=60, verbose=1
)

# -------- 4) Export --------
keras_path = os.path.join(MODEL_DIR, "fkd_cnn.keras")
model.save(keras_path)
tfjs.converters.save_keras_model(model, MODEL_DIR)
print(f"Saved TFJS model to {MODEL_DIR}/model.json")

# -------- 5) Minimal static web app --------
INDEX_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Facial Keypoints — CNN Demo</title>
<link rel="stylesheet" href="style.css"/>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js"></script>
</head><body>
<header><h1>Facial Keypoints — CNN (TF.js)</h1>
<p>Upload a face image (ideally 96×96). Inference runs in your browser.</p></header>
<section class="controls"><button id="loadModelBtn">Load Model</button>
<input type="file" id="imageInput" accept="image/*"/></section>
<section class="canvas-row">
<div><h3>Input</h3><canvas id="inputCanvas" width="96" height="96"></canvas></div>
<div><h3>Prediction</h3><canvas id="overlayCanvas" width="96" height="96"></canvas></div>
</section>
<section class="metrics"><p id="status">Status: idle</p></section>
<footer><p>Single CNN regressor → 30 coords. Trained with masked MSE.</p></footer>
<script src="app.js"></script></body></html>"""

STYLE_CSS = """:root{--bg:#0f172a;--card:#111827;--fg:#e5e7eb;--accent:#38bdf8}
*{box-sizing:border-box;font-family:ui-sans-serif,system-ui}body{margin:0;background:var(--bg);color:var(--fg)}
header,footer{padding:16px;text-align:center;background:#0b1220}.controls{display:flex;gap:12px;padding:16px;justify-content:center;flex-wrap:wrap}
button{padding:10px 14px;border-radius:10px;border:1px solid #1f2937;background:#111827;color:var(--fg);cursor:pointer}
button:hover{border-color:var(--accent)}.canvas-row{display:flex;gap:24px;justify-content:center;padding:16px;flex-wrap:wrap}
.canvas-row div{background:var(--card);padding:12px;border-radius:12px;box-shadow:0 8px 20px rgba(0,0,0,.35)}h3{margin:6px 0 12px 0;text-align:center}
.metrics{text-align:center;padding:8px 0 24px;color:#cbd5e1}
"""

APP_JS = """let model=null;
const inputCanvas=document.getElementById('inputCanvas');
const overlayCanvas=document.getElementById('overlayCanvas');
const ctxIn=inputCanvas.getContext('2d'); const ctxOut=overlayCanvas.getContext('2d');
const statusEl=document.getElementById('status');
document.getElementById('loadModelBtn').addEventListener('click',async()=>{
  statusEl.textContent='Loading model...';
  model = await tf.loadLayersModel('model/model.json');
  statusEl.textContent='Model loaded.';
});
document.getElementById('imageInput').addEventListener('change',async(e)=>{
  const file=e.target.files[0]; if(!file) return; await drawAndPredict(file);
});
async function drawAndPredict(file){
  if(!model){statusEl.textContent='Load the model first.';return;}
  const img=await fileToImage(file); ctxIn.clearRect(0,0,96,96); ctxIn.drawImage(img,0,0,96,96);
  const imgData=ctxIn.getImageData(0,0,96,96); const data=imgData.data; const gray=new Float32Array(96*96);
  for(let i=0,j=0;i<data.length;i+=4,j++){ gray[j]=(0.299*data[i]+0.587*data[i+1]+0.114*data[i+2])/255.0; }
  const t=tf.tensor(gray,[1,96,96,1]); const y=model.predict(t); const coords=(await y.array())[0]; y.dispose(); t.dispose();
  ctxOut.clearRect(0,0,96,96); ctxOut.drawImage(inputCanvas,0,0); ctxOut.strokeStyle='#38bdf8';
  for(let k=0;k<30;k+=2){ drawCross(ctxOut, coords[k], coords[k+1]); } statusEl.textContent='Prediction complete.';
}
function drawCross(ctx,x,y){ const r=2; ctx.beginPath(); ctx.moveTo(x-r,y); ctx.lineTo(x+r,y); ctx.moveTo(x,y-r); ctx.lineTo(x,y+r); ctx.stroke(); }
function fileToImage(file){ return new Promise((res)=>{ const img=new Image(); img.onload=()=>res(img); img.src=URL.createObjectURL(file); }); }
"""

with open(os.path.join(EXPORT_DIR,"index.html"),"w",encoding="utf-8") as f: f.write(INDEX_HTML)
with open(os.path.join(EXPORT_DIR,"style.css"),"w",encoding="utf-8") as f: f.write(STYLE_CSS)
with open(os.path.join(EXPORT_DIR,"app.js"),"w",encoding="utf-8") as f: f.write(APP_JS)

print(f"Web app scaffolded at: ./{EXPORT_DIR}")
print("Open web/index.html locally or host via GitHub Pages.")
