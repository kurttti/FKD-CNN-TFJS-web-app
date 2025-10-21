let model = null;
let modelLoadPromise = null;
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

const MODEL_URL = resolveAssetUrl("model/model.json?v=9");

const FALLBACK_BATCH_SHAPE = [
  null,
  inputCanvas.height || 96,
  inputCanvas.width || 96,
  1,
];

async function ensureModelLoaded() {
  if (model) {
    return model;
  }

  if (!modelLoadPromise) {
    modelLoadPromise = (async () => {
      const loadedModel = await loadModelWithPatchedInput();
      warmUpModel(loadedModel);
      return loadedModel;
    })();
  }

  try {
    model = await modelLoadPromise;
    return model;
  } catch (error) {
    modelLoadPromise = null;
    model = null;
    throw error;
  }
}

async function loadModelWithPatchedInput() {
  const handler = tf.io.browserHTTPRequest(MODEL_URL, {
    requestInit: { cache: "no-cache" },
  });
  const artifacts = await handler.load();
  ensureInputLayerBatchShape(artifacts);
  const memoryHandler = tf.io.fromMemory(artifacts);
  return tf.loadLayersModel(memoryHandler);
}

function ensureInputLayerBatchShape(artifacts) {
  const topology = artifacts?.modelTopology;
  if (!topology) {
    return;
  }

  const modelConfig = topology.model_config || topology.config;
  const config = modelConfig?.config || {};
  const layers = Array.isArray(config.layers) ? config.layers : [];

  layers
    .filter((layer) => layer?.class_name === "InputLayer")
    .forEach((layer) => {
      const layerConfig = layer.config || (layer.config = {});
      const resolvedShape = inferBatchShape(layerConfig, layers);
      layerConfig.batch_input_shape = resolvedShape;
      layerConfig.batch_shape = resolvedShape;
      layerConfig.batchInputShape = resolvedShape;
      layerConfig.batchShape = resolvedShape;
    });
}

function inferBatchShape(layerConfig, layers) {
  const existingShape =
    layerConfig.batch_input_shape ||
    layerConfig.batch_shape ||
    layerConfig.batchInputShape ||
    layerConfig.batchShape;

  if (Array.isArray(existingShape) && existingShape.length > 0) {
    return normalizeBatchShape(existingShape);
  }

  const inboundShape = extractInboundShape(layers);
  if (inboundShape) {
    return normalizeBatchShape(inboundShape);
  }

  return [...FALLBACK_BATCH_SHAPE];
}

function extractInboundShape(layers) {
  for (const layer of layers) {
    const inboundNodes = layer?.inbound_nodes;
    if (!Array.isArray(inboundNodes)) continue;
    for (const node of inboundNodes) {
      const args = Array.isArray(node?.args) ? node.args : [];
      for (const arg of args) {
        const shape = arg?.config?.shape;
        if (Array.isArray(shape) && shape.length > 0) {
          return [...shape];
        }
      }
    }
  }
  return null;
}

function normalizeBatchShape(shape) {
  return shape.map((dim, index) => {
    if (index === 0) {
      return null;
    }
    if (dim === undefined || dim === -1) {
      return FALLBACK_BATCH_SHAPE[index] ?? null;
    }
    return dim;
  });
}

function warmUpModel(loadedModel) {
  try {
    tf.tidy(() => {
      const zeros = tf.zeros([
        1,
        FALLBACK_BATCH_SHAPE[1],
        FALLBACK_BATCH_SHAPE[2],
        FALLBACK_BATCH_SHAPE[3],
      ]);
      const result = loadedModel.predict(zeros);
      if (Array.isArray(result)) {
        result.forEach((tensor) => tensor.dispose());
      } else if (result) {
        result.dispose();
      }
    });
  } catch (error) {
    console.warn("Model warm-up failed", error);
  }
}


// ✅ Автоматическая загрузка модели при открытии страницы
window.addEventListener("load", async () => {
  try {
    statusEl.textContent = "Loading model...";
    await ensureModelLoaded();
    statusEl.textContent = "Model loaded. Upload a face image.";
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Model loading failed: ${err.message}`;
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
    await ensureModelLoaded();
    statusEl.textContent = "Model loaded successfully!";
  } catch (e) {
    console.error(e);
    statusEl.textContent = `Error loading model: ${e.message}`;
  }
});

// ✅ При загрузке изображения — делаем предсказание
document.getElementById("imageInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  await drawAndPredict(file);
});

async function drawAndPredict(file) {
  try {
    statusEl.textContent = model
      ? "Processing image..."
      : "Waiting for model to load...";
    const activeModel = await ensureModelLoaded();
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
    const y = activeModel.predict(t);
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
    statusEl.textContent = `Error during prediction: ${error.message}`;
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
