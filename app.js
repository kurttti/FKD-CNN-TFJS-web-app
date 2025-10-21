// Self-invoking function to avoid leaking variables into the global scope.
(function () {
  // Prevent double initialisation if the script is included multiple times.
  if (window.__FKD_APP_INITIALIZED__) {
    console.warn(
      "Facial keypoints app already initialized; skipping duplicate script execution."
    );
    return;
  }
  window.__FKD_APP_INITIALIZED__ = true;

  // References to the loaded model and the promise that resolves when it finishes loading.
  var model = null;
  var modelPromise = null;

  // DOM elements used throughout the app.
  var inputCanvas = document.getElementById("inputCanvas");
  var overlayCanvas = document.getElementById("overlayCanvas");
  var ctxIn = inputCanvas.getContext("2d");
  var ctxOut = overlayCanvas.getContext("2d");
  var statusEl = document.getElementById("status");

  /**
   * Update the on‑screen status message.  If no status element exists this
   * function does nothing.
   *
   * @param {string} text The status message to display.
   */
  function updateStatus(text) {
    if (statusEl) {
      statusEl.textContent = text;
    }
  }

  /**
   * Resolve a relative asset URL against the current page’s location.  This
   * helper accounts for GitHub Pages serving the app from a subpath by
   * reconstructing an absolute URL based on location.pathname.
   *
   * @param {string} assetPath The path to the asset relative to the current directory.
   * @returns {string} A fully resolved URL.
   */
  function resolveAssetUrl(assetPath) {
    var locationObj = window.location;
    var origin =
      locationObj.origin || locationObj.protocol + "//" + locationObj.host;
    var pathname = locationObj.pathname || "/";

    // If the pathname ends with a slash we can append directly.
    if (pathname.charAt(pathname.length - 1) === "/") {
      return origin + pathname + assetPath;
    }
    // Otherwise strip off the file segment if present.
    var lastSlash = pathname.lastIndexOf("/");
    var lastSegment = pathname.substring(lastSlash + 1);
    var looksLikeFile = lastSegment.indexOf(".") !== -1;
    var basePath = looksLikeFile
      ? pathname.substring(0, lastSlash + 1)
      : pathname + "/";

    return origin + basePath + assetPath;
  }

  /**
   * Run a single dummy inference to warm the model’s weights and allocate
   * internal tensors.  This reduces latency for the first real prediction.
   *
   * @param {tf.LayersModel} loadedModel The model to warm.
   */
  function warmModel(loadedModel) {
    tf.tidy(function () {
      var warmupInput = tf.zeros([1, 96, 96, 1]);
      var prediction = loadedModel.predict(warmupInput);

      // Dispose of any returned tensors to avoid memory leaks.
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

  /**
   * Lazily load the CNN model from disk.  If the model is already loaded this
   * returns it immediately.  Otherwise it fetches `model/model.json` and
   * associated weight shards.  A cache‑busting query parameter is used to
   * avoid stale responses from GitHub’s CDN.  Any error resets the promise.
   *
   * @returns {Promise<tf.LayersModel>} A promise that resolves to the loaded model.
   */
  function fetchModel() {
    // If the model has already been instantiated simply return it.
    if (model) {
      return Promise.resolve(model);
    }
    // If a load is already in progress return the existing promise.
    if (!modelPromise) {
      var modelUrl = resolveAssetUrl("model/model.json?v=18");
      modelPromise = tf
        .loadLayersModel(modelUrl, { requestInit: { cache: "no-cache" } })
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

  /**
   * Draw the provided image file to the input canvas, perform greyscale
   * preprocessing, run inference through the CNN and then draw the predicted
   * keypoint crosses to the overlay canvas.
   *
   * @param {File} file An image file (PNG/JPEG) selected by the user.
   * @returns {Promise<void>} A promise that resolves when prediction is complete.
   */
  function drawAndPredict(file) {
    updateStatus("Processing image...");
    return fileToImage(file).then(function (img) {
      // Draw the image into the 96×96 input canvas.
      ctxIn.clearRect(0, 0, 96, 96);
      ctxIn.drawImage(img, 0, 0, 96, 96);
      // Convert to greyscale.
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
      if (
        !primaryTensor ||
        typeof primaryTensor.dataSync !== "function"
      ) {
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

  /**
   * Draw a small cross at the specified (x,y) coordinate on the provided
   * canvas context.  Used to mark keypoint positions.
   *
   * @param {CanvasRenderingContext2D} ctx The canvas context to draw on.
   * @param {number} x The x coordinate.
   * @param {number} y The y coordinate.
   */
  function drawCross(ctx, x, y) {
    var r = 2;
    ctx.beginPath();
    ctx.moveTo(x - r, y);
    ctx.lineTo(x + r, y);
    ctx.moveTo(x, y - r);
    ctx.lineTo(x, y + r);
    ctx.stroke();
  }

  /**
   * Convert an image File object into an HTMLImageElement.  This uses
   * Object URLs to avoid loading from disk multiple times.
   *
   * @param {File} file The image file selected by the user.
   * @returns {Promise<HTMLImageElement>} A promise that resolves with the loaded image.
   */
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

  // ---------------------------------------------------------------------------
  // Event wiring
  // ---------------------------------------------------------------------------

  // Pre‑load the model on page load to reduce latency when the user first uploads.
  window.addEventListener("load", function () {
    if (model) {
      return;
    }
    updateStatus("Loading model...");
    fetchModel()
      .then(function () {
        updateStatus("Model loaded. Upload a face image or start the camera.");
        // Render model and dataset statistics once the model has loaded.
        renderStats();
      })
      .catch(function (err) {
        console.error(err);
        updateStatus("Model loading failed. Check console.");
      });
  });

  // Manual model loading via the "Load Model" button.  This can be used if the
  // user wants to load the model ahead of uploading an image or capturing a photo.
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
          // Display statistics and charts after manual load completes.
          renderStats();
        })
        .catch(function (error) {
          console.error(error);
          updateStatus("Error loading model.");
        });
    });
  }

  // Listen for file uploads.  When a file is selected, ensure the model is
  // loaded and then run prediction on the selected image.
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

  // Webcam integration.  These variables hold references to the controls and the
  // live stream.  They are looked up once on initialization.
  var startCameraBtn = document.getElementById("startCameraBtn");
  var captureBtn = document.getElementById("captureBtn");
  var stopCameraBtn = document.getElementById("stopCameraBtn");
  var webcamVideo = document.getElementById("webcamVideo");
  var cameraContainer = document.getElementById("cameraContainer");
  var webcamStream = null;

  if (
    startCameraBtn &&
    captureBtn &&
    stopCameraBtn &&
    webcamVideo &&
    cameraContainer
  ) {
    // Request access to the user's webcam and display the live feed.
    startCameraBtn.addEventListener("click", async function () {
      // If a stream is already active just show the video and enable capture.
      if (webcamStream) {
        cameraContainer.style.display = "block";
        captureBtn.disabled = false;
        stopCameraBtn.style.display = "inline-block";
        return;
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        updateStatus("Webcam not supported in this browser.");
        return;
      }
      try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 96, height: 96 },
        });
        webcamVideo.srcObject = webcamStream;
        cameraContainer.style.display = "block";
        captureBtn.disabled = false;
        stopCameraBtn.style.display = "inline-block";
      } catch (err) {
        console.error(err);
        updateStatus("Unable to access the camera.");
      }
    });

    // Capture the current frame from the webcam and run it through the model.
    captureBtn.addEventListener("click", function () {
      if (!webcamStream) {
        updateStatus("Camera not started.");
        return;
      }
      // Draw the current frame to the input canvas.
      try {
        ctxIn.clearRect(0, 0, 96, 96);
        ctxIn.drawImage(webcamVideo, 0, 0, 96, 96);
      } catch (err) {
        console.error(err);
        updateStatus("Error capturing image from camera.");
        return;
      }
      // Convert the canvas to a Blob and treat it like an uploaded file.
      inputCanvas.toBlob(function (blob) {
        if (!blob) {
          updateStatus("Could not capture image.");
          return;
        }
        var file = new File([blob], "capture.png", { type: "image/png" });
        function runPrediction() {
          drawAndPredict(file).catch(function (error) {
            console.error("Prediction error:", error);
            updateStatus("Error during prediction.");
          });
        }
        if (!model) {
          updateStatus("Loading model...");
          fetchModel()
            .then(function () {
              runPrediction();
            })
            .catch(function (error) {
              console.error(error);
              updateStatus("Model loading failed. Check console.");
            });
        } else {
          runPrediction();
        }
      }, "image/png");
    });

    // Handle stopping the webcam.  This stops all tracks, hides the
    // camera container, disables capture and hides the stop button.
    stopCameraBtn.addEventListener("click", function () {
      if (webcamStream) {
        webcamStream.getTracks().forEach(function (track) {
          track.stop();
        });
        webcamStream = null;
      }
      webcamVideo.srcObject = null;
      cameraContainer.style.display = "none";
      captureBtn.disabled = true;
      stopCameraBtn.style.display = "none";
    });
  }
})();