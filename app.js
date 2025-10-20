let model = null;
const statusEl = document.getElementById('status');
const inputCanvas = document.getElementById('inputCanvas');
const overlayCanvas = document.getElementById('overlayCanvas');
const ctxIn = inputCanvas.getContext('2d');
const ctxOut = overlayCanvas.getContext('2d');

const MODEL_URL = 'model/model.json?v=2'; // cache-bust once

window.addEventListener('load', async () => {
  try {
    statusEl.textContent = 'Loading model…';
    model = await tf.loadLayersModel(MODEL_URL);
    model.predict(tf.zeros([1,96,96,1])).dispose(); // warm-up
    statusEl.textContent = 'Model loaded. Upload a face.';
  } catch (e) {
    console.error(e);
    statusEl.textContent = 'Failed to load model. Open DevTools → Network.';
  }
});

document.getElementById('loadModelBtn').addEventListener('click', async () => {
  if (model) { statusEl.textContent = 'Model already loaded.'; return; }
  window.dispatchEvent(new Event('load'));
});

document.getElementById('imageInput').addEventListener('change', async (e) => {
  const file = e.target.files[0]; if (!file) return;
  if (!model) { statusEl.textContent = 'Loading model…'; await window.dispatchEvent(new Event('load')); }
  await drawAndPredict(file);
  statusEl.textContent = 'Prediction complete.';
});

async function drawAndPredict(file){
  const img = await fileToImage(file);
  ctxIn.clearRect(0,0,96,96);
  ctxIn.drawImage(img, 0,0,96,96);

  const data = ctxIn.getImageData(0,0,96,96).data;
  const gray = new Float32Array(96*96);
  for (let i=0,j=0; i<data.length; i+=4, j++){
    gray[j] = (0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2]) / 255.0;
  }
  const t = tf.tensor(gray, [1,96,96,1]);
  const y = model.predict(t);
  const coords = (await y.array())[0];
  y.dispose(); t.dispose();

  ctxOut.clearRect(0,0,96,96);
  ctxOut.drawImage(inputCanvas, 0,0);
  ctxOut.strokeStyle = '#38bdf8';
  for (let k=0; k<30; k+=2) drawCross(ctxOut, coords[k], coords[k+1]);
}

function drawCross(ctx,x,y){ const r=2;
  ctx.beginPath(); ctx.moveTo(x-r,y); ctx.lineTo(x+r,y);
  ctx.moveTo(x,y-r); ctx.lineTo(x,y+r); ctx.stroke(); }

function fileToImage(file){
  return new Promise(res => { const img = new Image();
    img.onload = () => res(img);
    img.src = URL.createObjectURL(file);
  });
}
