(function () {
  if (window.__FKD_APP_INITIALIZED__) {
    console.warn("Facial keypoints app already initialized; skipping duplicate script execution.");
    return;
  }
  window.__FKD_APP_INITIALIZED__ = true;

  var model = null;
  var modelPromise = null;

  var inputCanvas = document.getElementById("inputCanvas");
  var overlayCanvas = document.getElementById("overlayCanvas");
  var ctxIn = inputCanvas.getContext("2d");
  var ctxOut = overlayCanvas.getContext("2d");
  var statusEl = document.getElementById("status");

  function updateStatus(text) {
    if (statusEl) {
      statusEl.textContent = text;
    }
  }

  function resolveAssetUrl(assetPath) {
    var locationObj = window.location;
    var origin =
      locationObj.origin || locationObj.protocol + "//" + locationObj.host;
    var pathname = locationObj.pathname || "/";

    if (pathname.charAt(pathname.length - 1) === "/") {
      return origin + pathname + assetPath;
    }

    var lastSlash = pathname.lastIndexOf("/");
    var lastSegment = pathname.substring(lastSlash + 1);
    var looksLikeFile = lastSegment.indexOf(".") !== -1;
    var basePath = looksLikeFile
      ? pathname.substring(0, lastSlash + 1)
      : pathname + "/";

    return origin + basePath + assetPath;
  }

  function warmModel(loadedModel) {
    tf.tidy(function () {
      var warmupInput = tf.zeros([1, 96, 96, 1]);
      var prediction = loadedModel.predict(warmupInput);

      if (Array.isArray(prediction)) {
        for (var i = 0; i < prediction.length; i++) {
          var tensor = prediction[i];
          if (tensor && typeof tensor.dispose === "function") {
            tensor.dispose();
          }
        }
      } else if (prediction && typeof prediction.dispose === "function") {
        prediction.dispose();
      }
    });
  }

  function fetchModel() {
    if (model) {
      return Promise.resolve(model);
    }

    if (!modelPromise) {
      var modelUrl = resolveAssetUrl("model/model.json?v=16");
      modelPromise = tf
        .loadLayersModel(modelUrl, {
          requestInit: { cache: "no-cache" },
        })
        .then(function (loadedModel) {
          warmModel(loadedModel);
          model = loadedModel;
          return model;
        })
        .catch(function (error) {
          modelPromise = null;
          throw error;
        });
    }

    return modelPromise;
  }

  window.addEventListener("load", function () {
    if (model) {
      return;
    }

    updateStatus("Loading model...");
    fetchModel()
      .then(function () {
        updateStatus("Model loaded. Upload a face image.");
      })
      .catch(function (err) {
        console.error(err);
        updateStatus("Model loading failed. Check console.");
      });
  });

  var loadBtn = document.getElementById("loadModelBtn");
  if (loadBtn) {
    loadBtn.addEventListener("click", function () {
      if (model) {
        updateStatus("Model already loaded.");
        return;
      }

      updateStatus("Loading model manually...");
      fetchModel()
        .then(function () {
          updateStatus("Model loaded successfully!");
        })
        .catch(function (error) {
          console.error(error);
          updateStatus("Error loading model.");
        });
    });
  }

  var imageInput = document.getElementById("imageInput");
  if (imageInput) {
    imageInput.addEventListener("change", function (event) {
      var files = event.target.files || [];
      var file = files[0];
      if (!file) {
        return;
      }

      function handlePrediction() {
        drawAndPredict(file).catch(function (error) {
          console.error("Prediction error:", error);
          updateStatus("Error during prediction.");
        });
      }

      if (!model) {
        updateStatus("Loading model...");
        fetchModel()
          .then(function () {
            handlePrediction();
          })
          .catch(function (error) {
            console.error(error);
            updateStatus("Model loading failed. Check console.");
          });
      } else {
        handlePrediction();
      }
    });
  }

  function drawAndPredict(file) {
    updateStatus("Processing image...");

    return fileToImage(file).then(function (img) {
      ctxIn.clearRect(0, 0, 96, 96);
      ctxIn.drawImage(img, 0, 0, 96, 96);

      var imgData = ctxIn.getImageData(0, 0, 96, 96).data;
      var gray = new Float32Array(96 * 96);
      var length = imgData.length;
      var j = 0;

      for (var i = 0; i < length; i += 4) {
        gray[j] =
          (0.299 * imgData[i] +
            0.587 * imgData[i + 1] +
            0.114 * imgData[i + 2]) /
          255;
        j += 1;
      }

      var inputTensor = tf.tensor(gray, [1, 96, 96, 1]);
      var prediction = model.predict(inputTensor);
      var primaryTensor;

      if (Array.isArray(prediction)) {
        primaryTensor = prediction[0];
        for (var idx = 1; idx < prediction.length; idx++) {
          var extraTensor = prediction[idx];
          if (extraTensor && typeof extraTensor.dispose === "function") {
            extraTensor.dispose();
          }
        }
      } else {
        primaryTensor = prediction;
      }

      if (!primaryTensor || typeof primaryTensor.dataSync !== "function") {
        inputTensor.dispose();
        if (primaryTensor && typeof primaryTensor.dispose === "function") {
          primaryTensor.dispose();
        }
        throw new Error("Model prediction did not return a tensor output.");
      }

      var coordsData = primaryTensor.dataSync();
      var coords = Array.prototype.slice.call(coordsData);

      inputTensor.dispose();
      if (primaryTensor && typeof primaryTensor.dispose === "function") {
        primaryTensor.dispose();
      }

      ctxOut.clearRect(0, 0, 96, 96);
      ctxOut.drawImage(inputCanvas, 0, 0);
      ctxOut.strokeStyle = "#38bdf8";

      for (var k = 0; k < 30; k += 2) {
        drawCross(ctxOut, coords[k], coords[k + 1]);
      }

      updateStatus("Prediction complete!");
    });
  }

  function drawCross(ctx, x, y) {
    var r = 2;
    ctx.beginPath();
    ctx.moveTo(x - r, y);
    ctx.lineTo(x + r, y);
    ctx.moveTo(x, y - r);
    ctx.lineTo(x, y + r);
    ctx.stroke();
  }

  function fileToImage(file) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () {
        URL.revokeObjectURL(img.src);
        resolve(img);
      };
      img.onerror = function () {
        URL.revokeObjectURL(img.src);
        reject(new Error("Unable to load the selected image."));
      };
      img.src = URL.createObjectURL(file);
    });
  }
})();
