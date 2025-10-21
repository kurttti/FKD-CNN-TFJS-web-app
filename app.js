let model = null;
let modelPromise = null;

const inputCanvas = document.getElementById("inputCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const ctxIn = inputCanvas.getContext("2d");
const ctxOut = overlayCanvas.getContext("2d");
const statusEl = document.getElementById("status");

function resolveAssetUrl(assetPath) {
  const { origin, pathname } = window.location;
  if (pathname.endsWith("/")) {
    return `${origin}${pathname}${assetPath}`;
  }
  const lastSlash = pathname.lastIndexOf("/");
  const lastSegment = pathname.slice(lastSlash + 1);
  const looksLikeFile = lastSegment.includes(".");
  const basePath = looksLikeFile
    ? pathname.slice(0, lastSlash + 1)
    : `${pathname}/`;
  return `${origin}${basePath}${assetPath}`;
}

const MODEL_URL = resolveAssetUrl("model/model.json?v=7");

const MODEL_URL = resolveAssetUrl("model/model.json?v=11");

async function fetchModel() {
  if (model) {
    return model;
  }

  if (!modelPromise) {
    modelPromise = (async () => {
      const loadedModel = await tf.loadLayersModel(MODEL_URL, {
        requestInit: { cache: "no-cache" },
      });

      tf.tidy(() => {
        const warmupInput = tf.zeros([1, 96, 96, 1]);
        const prediction = loadedModel.predict(warmupInput);
        if (Array.isArray(prediction)) {
          prediction.forEach((tensor) => tensor.dispose());
        } else if (prediction) {
          prediction.dispose();
        }
      });

      model = loadedModel;
      return model;
    })().catch((error) => {
      modelPromise = null;
      throw error;
    });
  }

  return modelPromise;
}

window.addEventListener("load", async () => {
  if (model) {
    return;
  }

  try {
    statusEl.textContent = "Loading model...";
    await fetchModel();
    statusEl.textContent = "Model loaded. Upload a face image.";
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Model loading failed: ${err.message}`;
  }

document.getElementById("loadModelBtn").addEventListener("click", async () => {
  if (model) {
    statusEl.textContent = "Model already loaded.";
    return;
  }

  try {
    statusEl.textContent = "Loading model manually...";
    await fetchModel();
    statusEl.textContent = "Model loaded successfully!";
  } catch (e) {
    console.error(e);
    statusEl.textContent = `Error loading model: ${e.message}`;
  }
});

document.getElementById("imageInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  try {
    if (!model) {
      statusEl.textContent = "Loading model...";
      await fetchModel();
    }

    await drawAndPredict(file);
  } catch (error) {
    console.error(error);
    statusEl.textContent = "Model loading failed. Check console.";
  }
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

  function fileToImage(file) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(img.src);
        resolve(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(img.src);
        reject(new Error("Unable to load the selected image."));
      };
      img.src = URL.createObjectURL(file);
    });
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
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(img.src);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(img.src);
      reject(new Error("Unable to load the selected image."));
    };
    img.src = URL.createObjectURL(file);
  });

  document.getElementById("loadModelBtn").addEventListener("click", async () => {
    if (model) {
      statusEl.textContent = "Model already loaded.";
      return;
    }

    try {
      statusEl.textContent = "Loading model manually...";
      await fetchModel();
      statusEl.textContent = "Model loaded successfully!";
    } catch (e) {
      console.error(e);
      statusEl.textContent = "Error loading model.";
    }
  });

  document.getElementById("imageInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
      if (!model) {
        statusEl.textContent = "Loading model...";
        await fetchModel();
      }

      await drawAndPredict(file);
    } catch (error) {
      console.error(error);
      statusEl.textContent = "Model loading failed. Check console.";
    }
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
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(img.src);
        resolve(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(img.src);
        reject(new Error("Unable to load the selected image."));
      };
      img.src = URL.createObjectURL(file);
    });
  }
})();
