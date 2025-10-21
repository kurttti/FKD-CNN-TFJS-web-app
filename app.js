let model = null;
const inputCanvas = document.getElementById("inputCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const ctxIn = inputCanvas.getContext("2d");
const ctxOut = overlayCanvas.getContext("2d");
const statusEl = document.getElementById("status");

const MODEL_URL = "model/model.json?v=2";

// ✅ Автоматическая загрузка модели при открытии страницы
window.addEventListener("load", async () => {
  try {
    statusEl.textContent = "Loading model...";
    model = await tf.loadLayersModel(MODEL_URL);
    model.predict(tf.zeros([1, 96, 96, 1])).dispose();
    statusEl.textContent = "Model loaded. Upload a face image.";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Model loading failed. Check console.";
  }
});

// ✅ Альтернатива: кнопка "Load Model"
document.getElementById("loadModelBtn").addEventListener("click", async () => {
  if (model) {
    statusEl.textContent = "Model already loaded.";
    return;
  }
  try {
    statusEl.textContent = "Loading model manually...";
    model = await tf.loadLayersModel(MODEL_URL);
    model.predict(tf.zeros([1, 96, 96, 1])).dispose();
    statusEl.textContent = "Model loaded successfully!";
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Error loading model.";
  }
});

// ✅ При загрузке изображения — делаем предсказание
document.getElementById("imageInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  if (!model) {
    statusEl.textContent = "Model not loaded yet.";
    return;
  }
  await drawAndPredict(file);
});

async function drawAndPredict(file) {
  try {
    statusEl.textContent = "Processing image...";
    const img = await fileToImage(file);
    ctxIn.clearRect(0, 0, 96, 96);
    ctxIn.drawImage(img, 0, 0, 96, 96);

    const imgData = ctxIn.getImageData(0, 0, 96, 96).data;
    const gray = new Float32Array(96 * 96);
    for (let i = 0, j = 0; i < imgData.length; i += 4, j++) {
      gray[j] =
        (0.299 * imgData[i] + 0.587 * imgData[i + 1] + 0.114 * imgData[i + 2]) /
        255.0;
    }

    const t = tf.tensor(gray, [1, 96, 96, 1]);
    const y = model.predict(t);
    const coords = (await y.array())[0];
    y.dispose();
    t.dispose();

    ctxOut.clearRect(0, 0, 96, 96);
    ctxOut.drawImage(inputCanvas, 0, 0);
    ctxOut.strokeStyle = "#38bdf8";
    for (let k = 0; k < 30; k += 2) drawCross(ctxOut, coords[k], coords[k + 1]);

    statusEl.textContent = "Prediction complete!";
  } catch (error) {
    console.error("Prediction error:", error);
    statusEl.textContent = "Error during prediction.";
  }
}

function drawCross(ctx, x, y) {
  const r = 2;
  ctx.beginPath();
  ctx.moveTo(x - r, y);
  ctx.lineTo(x + r, y);
  ctx.moveTo(x, y - r);
  ctx.lineTo(x, y + r);
  ctx.stroke();
}

function fileToImage(file) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = URL.createObjectURL(file);
  });
}
