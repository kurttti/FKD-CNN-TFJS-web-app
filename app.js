// Self-invoking function to avoid leaking variables into the global scope.
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

  // Display dataset and model statistics
  function renderStats() {
    var statsContainer = document.getElementById("statsContainer");
    if (!statsContainer) {
      return;
    }
    var statsList = document.getElementById("statsList");
    if (statsList) {
      var statItems = [
        { label: "Training images", value: 7049 },
        { label: "Test images", value: 1783 },
        { label: "Image size", value: "96×96 pixels" },
        { label: "Number of keypoints", value: 15 },
        { label: "Training epochs", value: 10 },
      ];
      statsList.innerHTML = "";
      statItems.forEach(function (item) {
        var li = document.createElement("li");
        li.textContent = item.label + ": " + item.value;
        statsList.appendChild(li);
      });
    }
    var lossCtx = document.getElementById("lossChart");
    if (lossCtx && typeof Chart !== "undefined") {
      var lossCanvasCtx = lossCtx.getContext("2d");
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
            x: { title: { display: true, text: "Epoch" } },
            y: {
              title: { display: true, text: "Loss" },
              beginAtZero: true,
            },
          },
        },
      });
    }

    var maeCtx = document.getElementById("maeChart");
    if (maeCtx && typeof Chart !== "undefined") {
      var maeCanvasCtx = maeCtx.getContext("2d");
      var keypointLabels = [
        "L eye center", "R eye center", "L eye inner", "L eye outer",
        "R eye inner", "R eye outer", "L brow inner", "L brow outer",
        "R brow inner", "R brow outer", "Nose tip",
        "Mouth left", "Mouth right", "Mouth top", "Mouth bottom"
      ];
      var keypointErrors = [2.5, 2.6, 2.3, 2.4, 2.6, 2.5, 2.2, 2.5, 2.3, 2.6, 3.0, 2.8, 2.8, 3.1, 3.2];
      new Chart(maeCanvasCtx, {
        type: "bar",
        data: {
          labels: keypointLabels,
          datasets: [{ label: "MAE (px)", data: keypointErrors, backgroundColor: "#38bdf8" }],
        },
        options: {
          responsive: false,
          maintainAspectRatio: false,
          scales: {
            x: {
              ticks: { maxRotation: 60, minRotation: 60, autoSkip: false },
            },
            y: {
              beginAtZero: true,
              title: { display: true, text: "Mean Absolute Error (pixels)" },
            },
          },
        },
      });
    }
    statsContainer.style.display = "block";
  }

  // Compute metrics from coords and display metrics + interpretation
  function computeAndDisplayMetrics(coords) {
    var metricsContainer = document.getElementById("photoMetrics");
    var list = document.getElementById("photoMetricsList");
    var interpEl = document.getElementById("photoInterpretation");
    if (!metricsContainer || !list || !Array.isArray(coords)) {
      return;
    }
    var lx = coords[0], ly = coords[1];
    var rx = coords[2], ry = coords[3];
    var noseX = coords[20], noseY = coords[21];
    var mouthLeftX = coords[22], mouthLeftY = coords[23];
    var mouthRightX = coords[24], mouthRightY = coords[25];
    var mouthTopX = coords[26], mouthTopY = coords[27];
    var mouthBottomX = coords[28], mouthBottomY = coords[29];

    var dxEye = rx - lx;
    var dyEye = ry - ly;
    var interocular = Math.sqrt(dxEye * dxEye + dyEye * dyEye);

    var dxMouth = mouthRightX - mouthLeftX;
    var dyMouth = mouthRightY - mouthLeftY;
    var mouthWidth = Math.sqrt(dxMouth * dxMouth + dyMouth * dyMouth);

    var ratio = mouthWidth !== 0 ? interocular / mouthWidth : 0;

    var mouthCenterX = (mouthTopX + mouthBottomX) / 2;
    var mouthCenterY = (mouthTopY + mouthBottomY) / 2;
    var dxNose = noseX - mouthCenterX;
    var dyNose = noseY - mouthCenterY;
    var noseToMouth = Math.sqrt(dxNose * dxNose + dyNose * dyNose);

    var items = [
      { label: "Inter‑ocular distance", value: interocular.toFixed(1) + " px" },
      { label: "Mouth width", value: mouthWidth.toFixed(1) + " px" },
      { label: "Eye/Mouth ratio", value: ratio.toFixed(2) },
      { label: "Nose to mouth distance", value: noseToMouth.toFixed(1) + " px" },
    ];
    list.innerHTML = "";
    items.forEach(function (item) {
      var li = document.createElement("li");
      li.textContent = item.label + ": " + item.value;
      list.appendChild(li);
    });

    // Interpretation logic
    if (interpEl) {
      var messageParts = [];
      var eyeYDiff = Math.abs(ly - ry);
      var midX = (lx + rx) / 2;
      var symDiff = Math.abs((midX - lx) - (rx - midX));

      if (interocular !== 0 && eyeYDiff < 3 && symDiff / interocular < 0.1) {
        messageParts.push("Facial symmetry appears balanced");
      } else {
        messageParts.push("There may be a slight asymmetry or head tilt");
      }

      if (ratio < 0.9) {
        messageParts.push("expression appears cheerful or smiling");
      } else if (ratio > 1.1) {
        messageParts.push("expression appears neutral");
      } else {
        messageParts.push("expression appears relaxed");
      }

      if (noseToMouth < 9) {
        messageParts.push("head may be tilted slightly upward");
      } else if (noseToMouth > 15) {
        messageParts.push("head may be tilted slightly downward");
      }

      var interpretation = messageParts.join("; ") + ".";
      interpEl.textContent = interpretation;
    }
    metricsContainer.style.display = "block";
  }

  function resolveAssetUrl(assetPath) {
    var loc = window.location;
    var origin = loc.origin || loc.protocol + "//" + loc.host;
    var pathname = loc.pathname || "/";
    if (pathname.charAt(pathname.length - 1) === "/") {
      return origin + pathname + assetPath;
    }
    var lastSlash = pathname.lastIndexOf("/");
    var lastSegment = pathname.substring(lastSlash + 1);
    var looksLikeFile = lastSegment.indexOf(".") !== -1;
    var basePath = looksLikeFile ? pathname.substring(0, lastSlash + 1) : pathname + "/";
    return origin + basePath + assetPath;
  }

  function warmModel(loadedModel) {
    tf.tidy(function () {
      var warmupInput = tf.zeros([1, 96, 96, 1]);
      var prediction = loadedModel.predict(warmupInput);
      if (Array.isArray(prediction)) {
        prediction.forEach(function (tensor) {
          if (tensor && typeof tensor.dispose === "function") {
            tensor.dispose();
          }
        });
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
      var modelUrl = resolveAssetUrl("model/model.json?v=20");
      modelPromise = tf.loadLayersModel(modelUrl, { requestInit: { cache: "no-cache" } })
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
        gray[j] = (0.299 * imgData[i] + 0.587 * imgData[i + 1] + 0.114 * imgData[i + 2]) / 255;
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
      computeAndDisplayMetrics(coords);
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

  window.addEventListener("load", function () {
    if (model) {
      return;
    }
    updateStatus("Loading model...");
    fetchModel()
      .then(function () {
        updateStatus("Model loaded. Upload a face image or start the camera.");
        renderStats();
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
          renderStats();
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
    startCameraBtn.addEventListener("click", async function () {
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

    captureBtn.addEventListener("click", function () {
      if (!webcamStream) {
        updateStatus("Camera not started.");
        return;
      }
      try {
        ctxIn.clearRect(0, 0, 96, 96);
        ctxIn.drawImage(webcamVideo, 0, 0, 96, 96);
      } catch (err) {
        console.error(err);
        updateStatus("Error capturing image from camera.");
        return;
      }
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
