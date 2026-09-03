/* Human evaluation participant UI backed by the local FastAPI service. */

const CONVERSATION_LIMIT_SECONDS = 120;
const MODEL_READY_TIMEOUT_MS = 210000;

let studyConfig = null;
let backend = null;
let state = null;
let activeMicStream = null;
let activeSocket = null;
let activeAudioContext = null;
let activeInputNode = null;
let activeProcessor = null;
let activeSilentGain = null;
let activePlaybackCursor = 0;
let activePlaybackSources = [];
let expectedSocketClose = false;
let timerHandle = null;
let modelWarmPromise = null;
let refreshSetupUi = null;

const mainContent = document.getElementById("mainContent");
const progressLabel = document.getElementById("progressLabel");
const progressBar = document.getElementById("progressBar");
const progressTrack = document.querySelector(".progress-track");
class StudyBackend {
  async request(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) }
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed (${response.status})`);
    }
    return response.json();
  }

  async createOrResumeSession() {
    const userId = sessionStorage.getItem("humanEvalUserId");
    const session = await this.request("/api/study-sessions", {
      method: "POST",
      body: JSON.stringify({ user_id: userId })
    });
    sessionStorage.setItem("humanEvalUserId", session.user_id);
    return mapServerSession(session);
  }

  warmModel() {
    return this.request("/api/model/readiness");
  }

  openConversation(conversation) {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    return new WebSocket(`${protocol}//${location.host}/api/conversations/${conversation.conversationId}/stream`);
  }

  finalizeConversation(conversation) {
    const abnormal = !["user_finished", "time_limit"].includes(conversation.endReason);
    return this.request(`/api/conversations/${conversation.conversationId}/finalize`, {
      method: "POST",
      body: JSON.stringify({
        end_reason: conversation.endReason,
        timeout: conversation.endReason === "time_limit",
        crash: conversation.endReason === "crash",
        disconnect: conversation.endReason === "disconnect",
        error: abnormal ? "Browser model stream ended unexpectedly" : null
      })
    });
  }

  submitConversationRating(conversation) {
    return this.request(`/api/conversations/${conversation.conversationId}/rating`, {
      method: "PUT",
      body: JSON.stringify({ metrics: conversation.ratings, feedback: conversation.feedback })
    });
  }

  submitPairComparison(task, comparison) {
    return this.request(`/api/tasks/${task.taskId}/comparison`, {
      method: "PUT",
      body: JSON.stringify({
        preference: comparison.preference,
        reasons: comparison.reasons,
        feedback: comparison.comment
      })
    });
  }

  async completeSession() {
    const result = await this.request(`/api/study-sessions/${state.sessionId}/complete`, {
      method: "POST"
    });
    state.completionCode = result.completion_code;
    return result.completion_code;
  }

  appendEvent() {}
  async appendRemoteEvent() {}
  persist() {}
}

function mapServerSession(session) {
  const tasks = session.tasks.map(task => {
    const conversations = task.conversations.map(conversation => ({
      conversationId: conversation.conversation_id,
      scenarioId: conversation.scenario_id,
      scenario: conversation.scenario,
      phase: conversation.rating || conversation.evaluation_status === "completed"
        ? "done"
        : ["interaction_completed", "completed", "failed", "abandoned"].includes(conversation.status)
          ? "rating"
          : "idle",
      liveStatus: "idle",
      startedAt: conversation.started_at,
      endedAt: conversation.ended_at,
      deadlineAt: null,
      endReason: conversation.end_reason,
      durationMs: null,
      targetTurns: conversation.target_turns || conversation.scenario?.targetTurns || (task.capability === "S3" ? 3 : 2),
      ratings: conversation.rating?.metrics || {},
      feedback: conversation.rating?.feedback || ""
    }));
    const firstUnrated = conversations.findIndex(conversation => conversation.phase !== "done");
    const comparison = task.comparison
      ? { preference: task.comparison.preference, reasons: task.comparison.reasons, comment: task.comparison.feedback }
      : { preference: null, reasons: [], comment: "" };
    return {
      taskId: task.task_id,
      taskKey: task.capability_key,
      capabilityCode: task.capability,
      title: task.title,
      substep: firstUnrated === -1 ? 2 : firstUnrated,
      complete: Boolean(task.comparison),
      conversations,
      comparison
    };
  });
  return {
    sessionId: session.session_id,
    completionCode: session.completion_code || null,
    currentMajor: session.status === "completed" ? "thanks" : "setup",
    setupComplete: false,
    device: { mic: false, speaker: false, model: false, modelError: false, consent: false },
    tasks
  };
}

async function init() {
  try {
    const response = await fetch("scenarios.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Unable to load the scenario configuration (${response.status})`);
    studyConfig = await response.json();
    backend = new StudyBackend();
    state = await backend.createOrResumeSession();
    document.getElementById("sessionId").textContent = `Participant ${state.sessionId}`;
    render();
    startModelWarmup();
  } catch (error) {
    mainContent.innerHTML = `<section class="surface rating-panel"><h1>Unable to initialize the page</h1><p class="lede">${escapeHtml(error.message)}</p><p class="muted">Start the FastAPI backend and open the URL it prints.</p></section>`;
  }
}

function render() {
  updateProgressChrome();
  if (state.currentMajor === "setup") return renderSetup();
  if (state.currentMajor.startsWith("task-")) {
    const taskIndex = Number(state.currentMajor.split("-")[1]);
    return renderTask(taskIndex);
  }
  return renderThanks();
}

function updateProgressChrome() {
  const taskIndex = state.currentMajor.startsWith("task-") ? Number(state.currentMajor.split("-")[1]) : null;
  let label = "Device check";
  let progress = 5;
  if (taskIndex !== null) {
    const task = state.tasks[taskIndex];
    const majorNumber = taskIndex + 2;
    const subNumber = task.substep + 1;
    const subLabels = ["Conversation 1", "Conversation 2", "Overall comparison"];
    label = `Task ${taskIndex + 1} · ${majorNumber}.${subNumber} ${subLabels[task.substep]}`;
    progress = taskIndex === 0 ? [22, 35, 48][task.substep] : [60, 73, 86][task.substep];
  } else if (state.currentMajor === "thanks") {
    label = "Study complete";
    progress = 100;
  }
  progressLabel.textContent = label;
  progressBar.style.width = `${progress}%`;
  progressTrack.setAttribute("aria-valuenow", String(progress));

  document.querySelectorAll(".major-tab").forEach((button, index) => {
    const key = button.dataset.major;
    const complete = index === 0 ? state.setupComplete : index === 1 ? state.tasks[0].complete : index === 2 ? state.tasks[1].complete : state.currentMajor === "thanks";
    const current = key === state.currentMajor;
    button.classList.toggle("is-current", current);
    button.classList.toggle("is-complete", complete && !current);
    button.toggleAttribute("disabled", !current);
    if (current) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
}

function renderSetup() {
  mainContent.replaceChildren(document.getElementById("setupTemplate").content.cloneNode(true));
  const micRow = mainContent.querySelector('[data-device="microphone"]');
  const speakerRow = mainContent.querySelector('[data-device="speaker"]');
  const modelRow = mainContent.querySelector('[data-device="model"]');
  const micStatus = document.getElementById("micStatus");
  const speakerStatus = document.getElementById("speakerStatus");
  const modelStatus = document.getElementById("modelStatus");
  const retryModel = document.getElementById("retryModel");
  const consent = document.getElementById("consentCheck");
  const enterButton = document.getElementById("enterStudy");

  function refreshDeviceUi() {
    micRow.classList.toggle("is-complete", state.device.mic);
    speakerRow.classList.toggle("is-complete", state.device.speaker);
    modelRow.classList.toggle("is-complete", state.device.model);
    micStatus.textContent = state.device.mic ? "Test passed" : "Connect your headset before testing";
    speakerStatus.textContent = state.device.speaker ? "Test sound played" : "Use headphones to prevent echo";
    modelStatus.textContent = state.device.model
      ? "Ready"
      : state.device.modelError
        ? "Unable to start the model"
        : "Starting model… the first load may take up to two minutes";
    retryModel.hidden = !state.device.modelError;
    consent.checked = state.device.consent;
    enterButton.disabled = !(state.device.mic && state.device.speaker && state.device.model && state.device.consent);
    enterButton.textContent = state.device.model ? "Continue to Task 1" : "Waiting for voice model…";
  }

  document.getElementById("testMic").addEventListener("click", async event => {
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = "Testing…";
    hideDeviceError();
    try {
      await runMicrophoneTest();
      state.device.mic = true;
      backend.appendEvent(state, "microphone_check_passed");
      backend.persist(state);
    } catch (error) {
      showDeviceError(`Unable to use the microphone: ${error.message}`);
    } finally {
      button.disabled = false;
      button.textContent = "Test again";
      refreshDeviceUi();
    }
  });

  document.getElementById("testSpeaker").addEventListener("click", async event => {
    const button = event.currentTarget;
    button.disabled = true;
    hideDeviceError();
    try {
      await playSpeakerTest();
      state.device.speaker = true;
      backend.appendEvent(state, "speaker_check_passed");
      backend.persist(state);
    } catch (error) {
      showDeviceError(`Unable to play the test sound: ${error.message}`);
    } finally {
      button.disabled = false;
      refreshDeviceUi();
    }
  });

  retryModel.addEventListener("click", startModelWarmup);

  consent.addEventListener("change", () => {
    state.device.consent = consent.checked;
    backend.appendEvent(state, "consent_changed", { accepted: consent.checked });
    backend.persist(state);
    refreshDeviceUi();
  });

  enterButton.addEventListener("click", async () => {
    state.setupComplete = true;
    const nextTask = state.tasks.findIndex(task => !task.complete);
    state.currentMajor = nextTask === -1 ? "thanks" : `task-${nextTask}`;
    await backend.appendRemoteEvent("setup_completed");
    render();
  });
  refreshSetupUi = refreshDeviceUi;
  refreshDeviceUi();
}

function startModelWarmup() {
  if (modelWarmPromise) return modelWarmPromise;
  state.device.model = false;
  state.device.modelError = false;
  refreshSetupUi?.();
  modelWarmPromise = backend.warmModel()
    .then(() => {
      state.device.model = true;
      state.device.modelError = false;
    })
    .catch(() => {
      state.device.model = false;
      state.device.modelError = true;
    })
    .finally(() => {
      modelWarmPromise = null;
      refreshSetupUi?.();
    });
  return modelWarmPromise;
}

async function runMicrophoneTest() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser does not support microphone access");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  const meterWrap = document.getElementById("micMeterWrap");
  const meter = document.getElementById("micMeter");
  meterWrap.hidden = false;
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  source.connect(analyser);
  const data = new Uint8Array(analyser.frequencyBinCount);
  const started = performance.now();

  await new Promise(resolve => {
    function sample() {
      analyser.getByteFrequencyData(data);
      const average = data.reduce((sum, value) => sum + value, 0) / data.length;
      meter.style.width = `${Math.min(100, average * 1.8)}%`;
      if (performance.now() - started < 3000) requestAnimationFrame(sample);
      else resolve();
    }
    sample();
  });
  stream.getTracks().forEach(track => track.stop());
  await audioContext.close();
  meterWrap.hidden = true;
}

async function playSpeakerTest() {
  const audioContext = new AudioContext();
  const gain = audioContext.createGain();
  gain.gain.value = 0.08;
  gain.connect(audioContext.destination);
  const now = audioContext.currentTime;
  [440, 660].forEach((frequency, index) => {
    const oscillator = audioContext.createOscillator();
    oscillator.frequency.value = frequency;
    oscillator.connect(gain);
    oscillator.start(now + index * 0.25);
    oscillator.stop(now + index * 0.25 + 0.18);
  });
  await simulateLatency(650);
  await audioContext.close();
}

function showDeviceError(message) {
  const error = document.getElementById("deviceError");
  error.textContent = message;
  error.hidden = false;
}

function hideDeviceError() {
  document.getElementById("deviceError").hidden = true;
}

function renderTask(taskIndex) {
  const task = state.tasks[taskIndex];
  const definition = getTaskDefinition(task.taskKey);
  mainContent.replaceChildren(document.getElementById("taskTemplate").content.cloneNode(true));
  const nav = mainContent.querySelector(".sub-tabs");
  const labels = ["Conversation 1 + rating", "Conversation 2 + rating", "Overall comparison"];
  labels.forEach((label, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sub-tab";
    button.textContent = `${taskIndex + 2}.${index + 1} ${label}`;
    button.classList.toggle("is-current", task.substep === index);
    button.classList.toggle("is-complete", task.substep > index || task.complete);
    button.disabled = task.substep !== index;
    if (task.substep === index) button.setAttribute("aria-current", "step");
    nav.appendChild(button);
  });

  if (task.substep < 2) {
    const conversation = task.conversations[task.substep];
    const scenario = conversation.scenario || definition.scenarios.find(item => item.id === conversation.scenarioId);
    if (conversation.phase === "idle") renderConversationIdle(taskIndex, task, conversation, scenario);
    else if (conversation.phase === "running") renderConversationRunning(taskIndex, task, conversation, scenario);
    else renderConversationRating(taskIndex, task, conversation, scenario);
  } else {
    renderPairComparison(taskIndex, task);
  }
}

function renderConversationIdle(taskIndex, task, conversation, scenario) {
  const stage = document.getElementById("taskStage");
  stage.innerHTML = `
    <div class="conversation-layout">
      <section class="conversation-panel surface">
        <div class="status-row"><span class="status-pill">${ordinalConversation(task.substep)}</span><span class="status-pill neutral">Not started</span></div>
        <h1>Start when you are ready</h1>
        <p class="lede">Read the task card first. When you start, the system will begin recording and start a two-minute countdown.</p>
        <div class="voice-stage">
          <div>
            ${voiceOrb(false)}
            <div class="voice-title">Start to talk</div>
            <p class="voice-copy">Use the numbered flow on the task card as your guide. Speak naturally, while keeping the same type of task and constraints.</p>
            <div class="voice-actions"><button id="startConversation" class="primary-button" type="button">Start conversation</button></div>
          </div>
        </div>
        ${conversationRuleTip(conversation.targetTurns)}
        <div id="conversationError" class="inline-error" role="alert" hidden></div>
      </section>
      ${renderTaskCard(scenario)}
    </div>`;

  document.getElementById("startConversation").addEventListener("click", () => startConversation(taskIndex, task.substep));
}

function renderConversationRunning(taskIndex, task, conversation, scenario) {
  const stage = document.getElementById("taskStage");
  stage.innerHTML = `
    <div class="conversation-layout">
      <section class="conversation-panel surface">
        <div class="status-row"><span class="status-pill live" id="livePill">Conversation in progress</span><span class="status-pill neutral">${ordinalConversation(task.substep)}</span></div>
        <h1>Conversation in progress</h1>
        <p class="lede">Use the task card as a guide and keep the conversation focused on its goal. You may finish whenever you feel you have tested it.</p>
        <div class="voice-stage">
          <div>
            ${voiceOrb(true)}
            <div class="voice-title" id="liveTitle">Listening</div>
            <p class="voice-copy" id="liveCopy">Keep speaking naturally. The assistant's response will play in the same voice session.</p>
            <div class="timer" id="conversationTimer">02:00</div>
            <div class="timer-label">Time remaining</div>
            <div class="voice-actions"><button id="finishConversation" class="danger-button" type="button">Finish the conversation</button></div>
          </div>
        </div>
        ${conversationRuleTip(conversation.targetTurns)}
      </section>
      ${renderTaskCard(scenario)}
    </div>`;

  document.getElementById("finishConversation").addEventListener("click", () => finishConversation(taskIndex, task.substep, "user_finished"));
  resumeTimer(taskIndex, task.substep);
  updateLiveStatus(conversation.liveStatus || "listening");
}

function renderConversationRating(taskIndex, task, conversation, scenario) {
  const stage = document.getElementById("taskStage");
  const ratingQuestions = studyConfig.ratingQuestions;
  stage.innerHTML = `
    <section class="rating-panel surface">
      <div class="status-row"><span class="status-pill">Interaction ended · rating pending</span><span class="status-pill neutral">${formatEndReason(conversation.endReason)}</span></div>
      <h1>Rate this conversation</h1>
      <p class="lede">1 means a serious failure and 5 means fully satisfied. Base your answers only on the conversation you just completed.</p>
      <div class="rating-list">
        ${ratingQuestions.map(question => renderRatingQuestion(question, conversation.ratings[question.id])).join("")}
      </div>
      <label class="comment-label" for="conversationFeedback">Optional comment</label>
      <textarea id="conversationFeedback" class="comment-box" placeholder="What worked well or failed in this conversation?">${escapeHtml(conversation.feedback)}</textarea>
      <div id="ratingError" class="form-error" role="alert" hidden>Complete all ratings before continuing.</div>
      <div class="form-footer"><button id="submitRating" class="primary-button" type="button">Save rating and continue</button></div>
    </section>`;

  stage.querySelectorAll(".score-button").forEach(button => button.addEventListener("click", () => {
    const questionId = button.dataset.question;
    conversation.ratings[questionId] = Number(button.dataset.score);
    stage.querySelectorAll(`[data-question="${questionId}"]`).forEach(peer => peer.setAttribute("aria-pressed", String(peer === button)));
    backend.persist(state);
  }));
  document.getElementById("conversationFeedback").addEventListener("input", event => {
    conversation.feedback = event.target.value;
  });
  document.getElementById("submitRating").addEventListener("click", () => submitConversationRating(taskIndex, task.substep, ratingQuestions));
}

function renderPairComparison(taskIndex, task) {
  const stage = document.getElementById("taskStage");
  const comparison = task.comparison;
  const reasons = ["About the same", "More accurate", "More complete", "Clearer", "Followed constraints better", "Used current information", "Did not require repetition", "More natural pacing"];
  stage.innerHTML = `
    <section class="comparison-panel surface">
      <div class="status-row"><span class="status-pill">Overall comparison</span></div>
      <h1>Considering only the response content, which conversation was better?</h1>
      <p class="lede">Compare the two conversations in this task. Model identities remain hidden.</p>
      <div class="choice-grid" role="group" aria-label="Overall preference">
        ${renderChoice("first", "Conversation 1 was better", "The first conversation in this task", comparison.preference)}
        ${renderChoice("second", "Conversation 2 was better", "The second conversation in this task", comparison.preference)}
        ${renderChoice("same", "About the same", "No clear preference", comparison.preference)}
      </div>
      <h2>Main reasons (select all that apply)</h2>
      <div class="reason-grid">${reasons.map(reason => `<button class="reason-button" type="button" data-reason="${reason}" aria-pressed="${comparison.reasons.includes(reason)}">${reason}</button>`).join("")}</div>
      <label class="comment-label" for="comparisonComment">Optional comment</label>
      <textarea id="comparisonComment" class="comment-box" placeholder="What influenced your choice?">${escapeHtml(comparison.comment)}</textarea>
      <div id="comparisonError" class="form-error" role="alert" hidden>Select an overall preference and at least one reason.</div>
      <div class="form-footer"><button id="submitComparison" class="primary-button" type="button">${taskIndex === 0 ? "Complete Task 1 and continue to Task 2" : "Complete Task 2"}</button></div>
    </section>`;

  stage.querySelectorAll(".choice-button").forEach(button => button.addEventListener("click", () => {
    comparison.preference = button.dataset.choice;
    stage.querySelectorAll(".choice-button").forEach(peer => peer.setAttribute("aria-pressed", String(peer === button)));
    backend.persist(state);
  }));
  stage.querySelectorAll(".reason-button").forEach(button => button.addEventListener("click", () => {
    const reason = button.dataset.reason;
    if (comparison.reasons.includes(reason)) comparison.reasons = comparison.reasons.filter(item => item !== reason);
    else comparison.reasons.push(reason);
    button.setAttribute("aria-pressed", String(comparison.reasons.includes(reason)));
    backend.persist(state);
  }));
  document.getElementById("comparisonComment").addEventListener("input", event => {
    comparison.comment = event.target.value;
    backend.persist(state);
  });
  document.getElementById("submitComparison").addEventListener("click", () => submitPairComparison(taskIndex));
}

function renderThanks() {
  mainContent.replaceChildren(document.getElementById("thanksTemplate").content.cloneNode(true));
  document.getElementById("completionCode").textContent = state.completionCode || "Complete";
}

async function startConversation(taskIndex, conversationIndex) {
  const conversation = state.tasks[taskIndex].conversations[conversationIndex];
  const errorBox = document.getElementById("conversationError");
  try {
    const button = document.getElementById("startConversation");
    button.disabled = true;
    button.textContent = "Starting model…";
    await startModelConversation(taskIndex, conversationIndex);
    conversation.phase = "running";
    conversation.liveStatus = "listening";
    conversation.startedAt = new Date().toISOString();
    conversation.deadlineAt = new Date(Date.now() + CONVERSATION_LIMIT_SECONDS * 1000).toISOString();
    render();
  } catch (error) {
    await stopConversationResources();
    errorBox.textContent = `Unable to start the conversation: ${error.message}`;
    errorBox.hidden = false;
    const button = document.getElementById("startConversation");
    if (button) {
      button.disabled = false;
      button.textContent = "Start conversation";
    }
  }
}

async function startModelConversation(taskIndex, conversationIndex) {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser does not support microphone access");
  activeMicStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  activeAudioContext = new AudioContext();
  await activeAudioContext.resume();
  activePlaybackCursor = activeAudioContext.currentTime;
  expectedSocketClose = false;

  const conversation = state.tasks[taskIndex].conversations[conversationIndex];
  const socket = backend.openConversation(conversation);
  activeSocket = socket;
  socket.addEventListener("message", event => handleModelMessage(event, taskIndex, conversationIndex));
  socket.addEventListener("close", () => {
    if (!expectedSocketClose && conversation.phase === "running") {
      finishConversation(taskIndex, conversationIndex, "disconnect");
    }
  });
  await new Promise((resolve, reject) => {
    const timeout = window.setTimeout(
      () => fail(new Error("The model service did not become ready within three minutes")),
      MODEL_READY_TIMEOUT_MS
    );
    const ready = event => {
      try {
        if (JSON.parse(event.data).type !== "ready") return;
      } catch {
        return;
      }
      cleanup();
      resolve();
    };
    const fail = error => {
      cleanup();
      reject(error instanceof Error ? error : new Error("Unable to connect to the model service"));
    };
    const cleanup = () => {
      clearTimeout(timeout);
      socket.removeEventListener("message", ready);
      socket.removeEventListener("error", fail);
      socket.removeEventListener("close", fail);
    };
    socket.addEventListener("message", ready);
    socket.addEventListener("error", fail);
    socket.addEventListener("close", fail);
  });

  activeInputNode = activeAudioContext.createMediaStreamSource(activeMicStream);
  activeProcessor = activeAudioContext.createScriptProcessor(2048, 1, 1);
  activeSilentGain = activeAudioContext.createGain();
  activeSilentGain.gain.value = 0;
  activeInputNode.connect(activeProcessor);
  activeProcessor.connect(activeSilentGain);
  activeSilentGain.connect(activeAudioContext.destination);
  const ratio = activeAudioContext.sampleRate / 16000;
  activeProcessor.onaudioprocess = event => {
    if (socket.readyState !== WebSocket.OPEN) return;
    const input = event.inputBuffer.getChannelData(0);
    const output = new Int16Array(Math.floor(input.length / ratio));
    for (let index = 0; index < output.length; index += 1) {
      const value = input[Math.floor(index * ratio)];
      output[index] = Math.max(-32768, Math.min(32767, value * 32767));
    }
    socket.send(output.buffer);
  };
}

function handleModelMessage(event, taskIndex, conversationIndex) {
  let message;
  try {
    message = JSON.parse(event.data);
  } catch {
    return;
  }
  // HUMAN_EVAL_DEBUG_REMOVE_AFTER_PILOT_BEGIN
  // Temporary model-arm visibility for pilot testing. Remove/disable before
  // collecting blinded participant ratings.
  if (message.type === "ready" && message.debug) {
    console.log("[HUMAN_EVAL_DEBUG_REMOVE_AFTER_PILOT] Current model config:", message.debug);
  }
  // HUMAN_EVAL_DEBUG_REMOVE_AFTER_PILOT_END
  if (message.type === "phase") {
    const status = message.v === "listening" ? "listening" : message.v === "answering" || message.v === "relaying" ? "speaking" : "processing";
    updateLiveStatus(status);
  } else if (message.type === "audio") {
    updateLiveStatus("speaking");
    playPcmAudio(message.pcm, message.sr || 24000);
  } else if (message.type === "turn") {
    updateLiveStatus("listening");
  } else if (message.type === "interrupt") {
    stopModelPlayback();
    updateLiveStatus("listening");
  } else if (message.type === "auto_finish") {
    finishConversation(taskIndex, conversationIndex, "time_limit");
  } else if (message.type === "error") {
    finishConversation(taskIndex, conversationIndex, "crash");
  }
}

function playPcmAudio(encodedPcm, sampleRate) {
  if (!activeAudioContext || !encodedPcm) return;
  const binary = atob(encodedPcm);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  const samples = new Int16Array(bytes.buffer);
  const buffer = activeAudioContext.createBuffer(1, samples.length, sampleRate);
  const channel = buffer.getChannelData(0);
  for (let index = 0; index < samples.length; index += 1) channel[index] = samples[index] / 32768;
  const source = activeAudioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(activeAudioContext.destination);
  activePlaybackCursor = Math.max(activePlaybackCursor, activeAudioContext.currentTime);
  source.start(activePlaybackCursor);
  activePlaybackCursor += buffer.duration;
  activePlaybackSources.push(source);
  source.addEventListener("ended", () => {
    activePlaybackSources = activePlaybackSources.filter(item => item !== source);
  });
}

function stopModelPlayback() {
  activePlaybackSources.forEach(source => {
    try { source.stop(); } catch { /* already stopped */ }
  });
  activePlaybackSources = [];
  activePlaybackCursor = activeAudioContext?.currentTime || 0;
}

function resumeTimer(taskIndex, conversationIndex) {
  clearInterval(timerHandle);
  const conversation = state.tasks[taskIndex].conversations[conversationIndex];
  const timer = document.getElementById("conversationTimer");

  function tick() {
    const remaining = Math.max(0, Math.ceil((new Date(conversation.deadlineAt).getTime() - Date.now()) / 1000));
    const minutes = Math.floor(remaining / 60);
    const seconds = remaining % 60;
    timer.textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
    timer.classList.toggle("is-low", remaining <= 20);
    if (remaining <= 0) finishConversation(taskIndex, conversationIndex, "time_limit");
  }
  tick();
  timerHandle = window.setInterval(tick, 250);
}

function updateLiveStatus(status) {
  const active = getActiveConversation();
  if (active) {
    active.liveStatus = status;
    backend.persist(state);
  }
  const title = document.getElementById("liveTitle");
  const copy = document.getElementById("liveCopy");
  if (!title || !copy) return;
  const labels = {
    listening: ["Listening", "You may add constraints, ask follow-up questions, or interrupt naturally."],
    processing: ["Processing", "The assistant is preparing a response."],
    speaking: ["Assistant is responding", "Keep listening or interrupt naturally."]
  };
  title.textContent = labels[status]?.[0] || labels.listening[0];
  copy.textContent = labels[status]?.[1] || labels.listening[1];
}

async function finishConversation(taskIndex, conversationIndex, reason) {
  const conversation = state.tasks[taskIndex].conversations[conversationIndex];
  if (conversation.phase !== "running") return;
  conversation.phase = "finishing";
  clearInterval(timerHandle);
  if (activeSocket?.readyState === WebSocket.OPEN) {
    activeSocket.send(JSON.stringify({ type: "finish_conversation" }));
  }
  await stopConversationResources(false);
  conversation.phase = "rating";
  conversation.liveStatus = "idle";
  conversation.endedAt = new Date().toISOString();
  conversation.endReason = reason;
  conversation.durationMs = new Date(conversation.endedAt).getTime() - new Date(conversation.startedAt).getTime();
  backend.persist(state);
  await backend.finalizeConversation(conversation);
  render();
}

async function submitConversationRating(taskIndex, conversationIndex, questions) {
  const task = state.tasks[taskIndex];
  const conversation = task.conversations[conversationIndex];
  const complete = questions.every(question => Number.isInteger(conversation.ratings[question.id]));
  if (!complete) {
    document.getElementById("ratingError").hidden = false;
    return;
  }
  await backend.submitConversationRating(conversation);
  conversation.phase = "done";
  task.substep = conversationIndex + 1;
  backend.persist(state);
  render();
}

async function submitPairComparison(taskIndex) {
  const task = state.tasks[taskIndex];
  const comparison = task.comparison;
  if (!comparison.preference || comparison.reasons.length === 0) {
    document.getElementById("comparisonError").hidden = false;
    return;
  }
  await backend.submitPairComparison(task, comparison);
  task.complete = true;
  if (taskIndex === 0) {
    state.currentMajor = "task-1";
  } else {
    state.currentMajor = "thanks";
    await backend.completeSession();
  }
  backend.persist(state);
  render();
}

async function stopConversationResources(closeSocket = true) {
  expectedSocketClose = true;
  clearInterval(timerHandle);
  activeProcessor?.disconnect();
  activeInputNode?.disconnect();
  activeSilentGain?.disconnect();
  activeMicStream?.getTracks().forEach(track => track.stop());
  stopModelPlayback();
  if (closeSocket && activeSocket && activeSocket.readyState < WebSocket.CLOSING) activeSocket.close();
  if (activeAudioContext && activeAudioContext.state !== "closed") await activeAudioContext.close();
  activeSocket = null;
  activeMicStream = null;
  activeAudioContext = null;
  activeInputNode = null;
  activeProcessor = null;
  activeSilentGain = null;
  activePlaybackCursor = 0;
}

function getActiveConversation() {
  if (!state?.currentMajor?.startsWith("task-")) return null;
  const taskIndex = Number(state.currentMajor.split("-")[1]);
  const task = state.tasks[taskIndex];
  return task.substep < 2 ? task.conversations[task.substep] : null;
}

function getTaskDefinition(taskKey) {
  return studyConfig.tasks.find(item => item.key === taskKey);
}

function renderTaskCard(scenario) {
  return `
    <aside class="task-card surface">
      <div class="status-row"><span class="status-pill">Task card</span></div>
      <h2>${escapeHtml(scenario.participantTitle)}</h2>
      <h3>Your goal</h3>
      <p class="muted">${escapeHtml(scenario.goal)}</p>
      <h3>Suggested conversation flow</h3>
      <ol>${scenario.prompts.map((prompt, index) => `<li><span>${index + 1}</span><div><strong>${escapeHtml(prompt.label)}</strong>${escapeHtml(prompt.text)}</div></li>`).join("")}</ol>
      <div class="task-instruction"><strong>Before you finish:</strong> ${escapeHtml(scenario.doNotRepeatInstruction)}</div>
    </aside>`;
}

function renderRatingQuestion(question, selected) {
  return `
    <div class="rating-question">
      <p><strong>${escapeHtml(question.label)}</strong><br>${escapeHtml(question.text)}</p>
      <div class="score-row" role="group" aria-label="${escapeHtml(question.text)}">
        ${[1, 2, 3, 4, 5].map(score => `<button class="score-button" type="button" data-question="${question.id}" data-score="${score}" aria-pressed="${selected === score}">${score}</button>`).join("")}
      </div>
      <div class="score-help"><span>Serious failure</span><span>Fully satisfied</span></div>
    </div>`;
}

function renderChoice(value, label, detail, selected) {
  return `<button class="choice-button" type="button" data-choice="${value}" aria-pressed="${selected === value}"><strong>${label}</strong><span>${detail}</span></button>`;
}

function voiceOrb(live) {
  return `<div class="voice-orb${live ? " is-live" : ""}" aria-hidden="true"><div class="orb-bars"><span></span><span></span><span></span><span></span><span></span></div></div>`;
}

function conversationRuleTip(targetTurns = 2) {
  return `<div class="rule-tip"><span class="tip-dot" aria-hidden="true"></span><span>Aim for about ${targetTurns} turns and wait for each answer before continuing. This is guidance, not a requirement—you may finish whenever the task feels complete. The conversation ends automatically after two minutes.</span></div>`;
}

function ordinalConversation(index) {
  return index === 0 ? "Conversation 1" : "Conversation 2";
}

function formatEndReason(reason) {
  if (reason === "time_limit") return "Ended automatically at the two-minute limit";
  if (reason === "crash") return "Ended because the model connection failed";
  if (reason === "disconnect") return "Ended because the connection closed";
  return "Ended by participant";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function simulateLatency(ms) {
  return new Promise(resolve => window.setTimeout(resolve, ms));
}

window.addEventListener("beforeunload", () => stopConversationResources());
document.addEventListener("DOMContentLoaded", init);
