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
   * Display dataset and model statistics and render charts using Chart.js.
   * This is called once the model has successfully loaded.
   */
  function renderStats() {
    var statsContainer = document.getElementById("statsContainer");
    if (!statsContainer) {
      return;
    }
    var statsList = document.getElementById("statsList");
    if (statsList) {
      // Populate dataset summary information. Replace the numbers below with real
      // statistics from your training process if available.
      var statItems = [
        { label: "Training images", value: 7049 },
        { label: "Test images", value: 1783 },
        { label: "Image size", value: "96×96 pixels" },
        { label: "Number of keypoints", value: 15 },
        { label: "Training epochs", value: 10 },
      ];
      // Clear any existing content.
      statsList.innerHTML = "";
      statItems.forEach(function (item) {
        var li = document.createElement("li");
        li.textContent = item.label + ": " + item.value;
        statsList.appendChild(li);
      });
    }
    // Render training vs validation loss line chart.
    var lossCtx = document.getElementById("lossChart");
    if (lossCtx && typeof Chart !== "undefined") {
      var lossCanvasCtx = lossCtx.getContext("2d");
      // Dummy loss values. Replace these with actual loss values if available.
      var trainingLoss = [0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045, 0.04];
      var valLoss = [0.17, 0.14, 0.13, 0.11, 0.10, 0.09, 0.085, 0.08, 0.075, 0.07];
      var epochs = trainingLoss.map(function (_, idx) {
        return idx + 1;
      });
      new Chart(lossCanvasCtx, {
        type: "line",
        data: {
          labels: epochs,
          datasets: [
            {
              label: "Training Loss",
              data: trainingLoss,
              borderColor: "#38bdf8",
              backgroundColor: "rgba(56, 189, 248, 0.2)",
              fill: false,
            },
            {
              label: "Validation Loss",
              data: valLoss,
              borderColor: "#fb7185",
              backgroundColor: "rgba(251, 113, 133, 0.2)",
              fill: false,
            },
          ],
        },
        options: {
          responsive: false,
          maintainAspectRatio: false,
          scales: {
            x: {
              title: {
                display: true,
                text: "Epoch",
              },
            },
            y: {
              title: {
                display: true,
                text: "Loss",
              },
              beginAtZero: true,
            },
          },
        },
      });
    }
    // Render per‑keypoint MAE bar chart.
    var maeCtx = document.getElementById("maeChart");
    if (maeCtx && typeof Chart !== "undefined") {
      var maeCanvasCtx = maeCtx.getContext("2d");
      // Dummy mean absolute error values (in pixel units). Replace these with real
      // metrics if available.
      var keypointLabels = [
        "L eye center",
        "R eye center",
        "L eye inner",
        "L eye outer",
        "R eye inner",
        "R eye outer",
        "L brow inner",
        "L brow outer",
        "R brow inner",
        "R brow outer",
        "Nose tip",
        "Mouth left",
        "Mouth right",
        "Mouth top",
        "Mouth bottom",
      ];
      var keypointErrors = [
        2.5, 2.6, 2.3, 2.4, 2.6, 2.5, 2.2, 2.5, 2.3, 2.6, 3.0, 2.8, 2.8, 3.1,
        3.2,
      ];
      new Chart(maeCanvasCtx, {
        type: "bar",
        data: {
          labels: keypointLabels,
          datasets: [
            {
              label: "MAE (px)",
              data: keypointErrors,
              backgroundColor: "#38bdf8",
            },
          ],
        },
        options: {
          responsive: false,
          maintainAspectRatio: false,
          scales: {
            x: {
              ticks: {
                maxRotation: 60,
                minRotation: 60,
                autoSkip: false,
              },
            },
            y: {
              beginAtZero: true,
              title: {
                display: true,
                text: "Mean Absolute Error (pixels)",
              },
            },
          },
        },
      });
    }
    // Make the stats container visible.
    statsContainer.style.display = "block";
  }

  /**
   * Compute simple measurements from the predicted keypoint coordinates and
   * display them in the Photo Analysis section.  Metrics include
   * inter‑ocular distance, mouth width, the ratio between them, and the
   * distance from the nose tip to the midpoint of the mouth.  All distances
   * are reported in pixel units based on the 96×96 input resolution.
   *
   * @param {number[]} coords Array of 30 numbers representing the x and y
   * coordinates of the 15 keypoints in the order defined by the dataset.
   */
  function computeAndDisplayMetrics(coords) {
    var metricsContainer = document.getElementById("photoMetrics");
    var list = document.getElementById("photoMetricsList");
    if (!metricsContainer || !list || !Array.isArray(coords)) {
      return;
    }
    // Extract key points based on dataset order.
    var lx = coords[0], ly = coords[1]; // left eye center
    var rx = coords[2], ry = coords[3]; // right eye center
    var noseX = coords[20], noseY = coords[21]; // nose tip
    var mouthLeftX = coords[22], mouthLeftY = coords[23]; // mouth left corner
    var mouthRightX = coords[24], mouthRightY = coords[25]; // mouth right corner
    var mouthTopX = coords[26], mouthTopY = coords[27]; // mouth center top lip
    var mouthBottomX = coords[28], mouthBottomY = coords[29]; // mouth center bottom lip
    // Inter‑ocular distance.
    var dxEye = rx - lx;
    var dyEye = ry - ly;
    var interocular = Math.sqrt(dxEye * dxEye + dyEye * dyEye);
    // Mouth width.
    var dxMouth = mouthRightX - mouthLeftX;
    var dyMouth = mouthRightY - mouthLeftY;
    var mouthWidth = Math.sqrt(dxMouth * dxMouth + dyMouth * dyMouth);
    // Ratio of eye distance to mouth width (aspect ratio). Avoid division by zero.
    var ratio = mouthWidth !== 0 ? interocular / mouthWidth : 0;
    // Nose to mouth distance (nose tip to midpoint of mouth top and bottom).
    var mouthCenterX = (mouthTopX + mouthBottomX) / 2;
    var mouthCenterY = (mouthTopY + mouthBottomY) / 2;
    var dxNose = noseX - mouthCenterX;
    var dyNose = noseY - mouthCenterY;
    var noseToMouth = Math.sqrt(dxNose * dxNose + dyNose * dyNose);
    // Populate the list with the computed metrics.
    var items = [
      {
        label: "Inter‑ocular distance",
        value: interocular.toFixed(1) + " px",
      },
      { label: "Mouth width", value: mouthWidth.toFixed(1) + " px" },
      {
        label: "Eye/Mouth ratio",
        value: ratio.toFixed(2),
      },
      {
        label: "Nose to mouth distance",
        value: noseToMouth.toFixed(1) + " px",
      },
    ];
    // Clear previous metrics and render the new values.
    list.innerHTML = "";
    items.forEach(function (item) {
      var li = document.createElement("li");
      li.textContent = item.label + ": " + item.value;
      list.appendChild(li);
    });
    // Show the metrics container.
    metricsContainer.style.display = "block";
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
      var modelUrl = resolveAssetUrl("model/model.json?v=19");
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
      // Update status and compute per‑photo metrics for the user.
      updateStatus("Prediction complete!");
      computeAndDisplayMetrics(coords);
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