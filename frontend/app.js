"use strict";

(function () {
  var form = document.getElementById("sentiment-form");
  var input = document.getElementById("text-input");
  var analyzeBtn = document.getElementById("analyze-btn");
  var spinner = analyzeBtn.querySelector(".spinner");
  var btnLabel = analyzeBtn.querySelector(".btn-label");
  var charCount = document.getElementById("char-count");
  var errorBox = document.getElementById("error-box");
  var resultBox = document.getElementById("result");

  var MAX_CHARS = 2000;

  var API_BASE_URL = (window.SENTIMENT_API_BASE || "http://localhost:8000").replace(/\/+$/, "");
  var API_ENDPOINT = API_BASE_URL + "/api/predict";

  function renderLabel(label) {
    var cls = "badge badge--" + label;
    var glyph = { positive: "\uD83D\uDC4D", negative: "\uD83D\uDC4E", neutral: "\uD83D\uDC4C" }[label] || "";
    return (glyph + " " + label.charAt(0).toUpperCase() + label.slice(1)).trim();
  }

  function renderBars(scores) {
    var order = ["negative", "neutral", "positive"];
    return order
      .map(function (label) {
        var pct = Math.round((scores[label] || 0) * 100);
        return (
          '<div class="bar">' +
          '<span class="bar__label">' + label + "</span>" +
          '<span class="bar__track"><span class="bar__fill bar__fill--' + label + '" style="width:' + pct + '%"></span></span>' +
          '<span class="bar__value">' + (pct / 100).toFixed(3) + "</span>" +
          "</div>"
        );
      })
      .join("");
  }

  function renderResult(data) {
    resultBox.innerHTML =
      '<h2 class="result__heading">Result</h2>' +
      '<div class="result__summary">' +
      '<span class="' + (data.label === "positive" ? "badge badge--positive" : data.label === "negative" ? "badge badge--negative" : "badge badge--neutral") + '">' + renderLabel(data.label) + "</span>" +
      '<span class="confidence">Confidence <strong>' + (data.confidence * 100).toFixed(1) + "%</strong></span>" +
      "</div>" +
      '<div class="bars">' + renderBars(data.scores) + "</div>";
    resultBox.hidden = false;
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.hidden = true;
  }

  function setLoading(loading) {
    analyzeBtn.disabled = loading;
    btnLabel.textContent = loading ? "Analyzing" : "Analyze";
    spinner.hidden = !loading;
  }

  function updateCharCount() {
    charCount.textContent = input.value.length + " / " + MAX_CHARS;
    analyzeBtn.disabled = input.value.trim().length === 0;
  }

  function withTimeout(ms) {
    var controller = new AbortController();
    var handle = setTimeout(function () {
      controller.abort();
    }, ms);
    return {
      signal: controller.signal,
      cancel: function () {
        clearTimeout(handle);
      },
    };
  }

  input.addEventListener("input", function () {
    clearError();
    updateCharCount();
    if (input.value.length > MAX_CHARS) {
      input.value = input.value.slice(0, MAX_CHARS);
      updateCharCount();
    }
  });

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var text = input.value.trim();
    if (!text) {
      showError("Please enter some text to analyze.");
      return;
    }

    clearError();
    resultBox.hidden = true;
    setLoading(true);

    var timeout = withTimeout(60000);

    fetch(API_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
      signal: timeout.signal,
    })
      .then(function (response) {
        return response.text().then(function (raw) {
          var data = null;
          try {
            data = JSON.parse(raw);
          } catch (e) {
            data = null;
          }
          return { ok: response.ok, status: response.status, data: data };
        });
      })
      .then(function (result) {
        if (result.ok && result.data !== null) {
          renderResult(result.data);
          return;
        }
        if (result.data === null) {
          showError("The service returned an unexpected response. Please try again.");
          return;
        }
        var detail = result.data.detail;
        if (typeof detail === "string") {
          showError(detail);
        } else if (result.status === 422) {
          showError("The text was rejected. Please shorten it and try again.");
        } else if (result.status >= 500) {
          showError("The model service is having trouble right now. Please try again later.");
        } else {
          showError("The service rejected the request (" + result.status + "). Please try again.");
        }
      })
      .catch(function (err) {
        if (err && err.name === "AbortError") {
          showError("The service took too long to respond. Please try again.");
        } else {
          showError("Network error — could not reach the service. Please try again.");
        }
      })
      .finally(function () {
        timeout.cancel();
        setLoading(false);
      });
  });

  updateCharCount();
})();