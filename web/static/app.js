let sessionId = null;
let recorder = null;
let playback = null;
let activeStream = null;
let ttsAbort = null;
let busy = false;
let turnCount = 0;
let memoryEventCount = 0;
let audioContext = null;
let silenceMonitor = null;
let recordingStartedAt = 0;
let speechStarted = false;
let lastVoiceAt = 0;
let recordingHadSpeech = false;
let recordingStopReason = "manual";
let voiceSocket = null;
let streamedTranscript = null;
let streamedTranscriptResolve = null;
let streamedTranscriptReject = null;
let conversationActive = false;
let currentPhase = "ready";
let bargeInStream = null;
let bargeInAudioContext = null;
let bargeInMonitor = null;
let bargeInTriggered = false;
let voiceStreamHandlers = {};
let streamedPlaybackChain = Promise.resolve();

const silenceThreshold = 0.018;
const bargeInThreshold = 0.035;
const bargeInFramesNeeded = 4;
const pauseToSubmitMs = 950;
const noSpeechTimeoutMs = 6000;
const maxRecordingMs = 45000;

const statusEl = document.querySelector("#status");
const phaseText = document.querySelector("#phaseText");
const conversationEl = document.querySelector("#conversation");
const historyCountEl = document.querySelector("#historyCount");
const memoryActivityEl = document.querySelector("#memoryActivity");
const memoryCountEl = document.querySelector("#memoryCount");
const conversationListEl = document.querySelector("#conversationList");
const newConversationButton = document.querySelector("#newConversationButton");
const micButton = document.querySelector("#micButton");
const micIcon = document.querySelector("#micIcon");
const stopButton = document.querySelector("#stopButton");
const voiceStage = document.querySelector(".voice-stage");
const speakerOutput = document.querySelector("#speakerOutput");

function setStatus(text, phase = "ready") {
  currentPhase = phase;
  statusEl.textContent = text;
  phaseText.textContent = text;
  voiceStage.dataset.phase = phase;
}

function addTurn(role, text, meta = {}) {
  if (!text && !meta.allowEmpty) return;
  const turn = document.createElement("article");
  turn.className = `turn turn-${role.toLowerCase()}`;
  turn.innerHTML = `<div class="turn-head"><strong></strong><span></span></div><p></p>`;
  turn.querySelector("strong").textContent = role;
  turn.querySelector("span").textContent = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  turn.querySelector("p").textContent = text;
  if (meta.memoryStored || meta.memoriesUsed?.length) {
    turn.append(memoryBadges(meta));
  }
  conversationEl.append(turn);
  conversationEl.scrollTop = conversationEl.scrollHeight;
  turnCount += 1;
  historyCountEl.textContent = `${turnCount} ${turnCount === 1 ? "turn" : "turns"}`;
  return turn;
}

function resetCurrentConversation() {
  conversationEl.innerHTML = "";
  turnCount = 0;
  historyCountEl.textContent = "0 turns";
}

function setTurnText(turn, text) {
  if (!turn) return;
  turn.querySelector("p").textContent = text;
  conversationEl.scrollTop = conversationEl.scrollHeight;
}

function memoryBadges(meta) {
  const row = document.createElement("div");
  row.className = "memory-badges";
  if (meta.memoryStored) {
    const stored = document.createElement("span");
    stored.className = "memory-chip memory-chip-new";
    stored.textContent = "Memory saved";
    row.append(stored);
  }
  for (const memory of meta.memoriesUsed || []) {
    const chip = document.createElement("span");
    chip.className = "memory-chip";
    chip.textContent = `Used: ${memory}`;
    row.append(chip);
  }
  return row;
}

function addMemoryActivity({ memoryStored = false, memoriesUsed = [] }) {
  if (!memoryStored && memoriesUsed.length === 0) {
    return;
  }

  const event = document.createElement("article");
  event.className = "memory-event";
  const title = memoryStored ? "Saved new memory" : "Used memory";
  event.innerHTML = `<strong></strong><ul></ul>`;
  event.querySelector("strong").textContent = title;
  const list = event.querySelector("ul");

  if (memoryStored) {
    const item = document.createElement("li");
    item.textContent = "The assistant marked this exchange as durable memory.";
    list.append(item);
  }
  for (const memory of memoriesUsed) {
    const item = document.createElement("li");
    item.textContent = memory;
    list.append(item);
  }

  memoryActivityEl.prepend(event);
  memoryEventCount += 1;
  memoryCountEl.textContent = `${memoryEventCount} ${memoryEventCount === 1 ? "event" : "events"}`;
}

async function sendText(text) {
  if (voiceSocket?.readyState === WebSocket.OPEN) {
    await sendTextStreaming(text);
    return;
  }
  busy = true;
  micButton.disabled = true;
  addTurn("You", text);
  setStatus("Thinking", "processing");
  try {
    const data = await postJson("/api/chat", { session_id: sessionId, text });
    sessionId = data.session_id;
    const memoriesUsed = Array.isArray(data.memories_used) ? data.memories_used : [];
    const memoryStored = Boolean(data.memory_stored);
    addTurn("Companion", data.text, { memoryStored, memoriesUsed });
    addMemoryActivity({ memoryStored, memoriesUsed });
    setStatus("Preparing voice", "processing");
    await playTts(data.text);
    await loadConversationList();
    setStatus(conversationActive ? "Listening again" : "Ready", "ready");
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    micButton.disabled = false;
    micIcon.textContent = "Mic";
    if (conversationActive) {
      setTimeout(() => {
        if (conversationActive && !busy && (!recorder || recorder.state !== "recording")) {
          startListening().catch((error) => showError(error, "Microphone error"));
        }
      }, 250);
    }
  }
}

async function sendTextStreaming(text) {
  busy = true;
  micButton.disabled = true;
  addTurn("You", text);
  const companionTurn = addTurn("Companion", "", { allowEmpty: true });
  let assistantText = "";
  let finalMeta = { memoryStored: false, memoriesUsed: [] };
  streamedPlaybackChain = Promise.resolve();
  setStatus("Thinking", "processing");

  try {
    await new Promise((resolve, reject) => {
      voiceStreamHandlers = {
        assistant_start: () => setStatus("Thinking", "processing"),
        assistant_delta: (payload) => {
          assistantText += payload.text || "";
          setTurnText(companionTurn, assistantText);
          setStatus("Speaking as thoughts arrive", "speaking");
        },
        assistant_audio: (payload) => {
          streamedPlaybackChain = streamedPlaybackChain.then(() => playAudioChunk(payload));
        },
        tts_error: (payload) => {
          addTurn("System", `Local TTS chunk failed. ${payload.message || ""}`);
        },
        assistant_done: (payload) => {
          finalMeta = {
            memoryStored: Boolean(payload.memory_stored),
            memoriesUsed: Array.isArray(payload.memories_used) ? payload.memories_used : [],
          };
          if (payload.text) {
            assistantText = payload.text;
            setTurnText(companionTurn, assistantText);
          }
          resolve();
        },
        error: (payload) => reject(new Error(payload.message || "Voice stream error")),
      };
      voiceSocket.send(JSON.stringify({ type: "text", text }));
    });
    await streamedPlaybackChain;
    const badges = memoryBadges(finalMeta);
    if (badges.children.length) companionTurn.append(badges);
    addMemoryActivity(finalMeta);
    await loadConversationList();
    setStatus(conversationActive ? "Listening again" : "Ready", "ready");
  } catch (error) {
    showError(error);
  } finally {
    voiceStreamHandlers = {};
    busy = false;
    micButton.disabled = false;
    micIcon.textContent = "Mic";
    if (conversationActive) {
      setTimeout(() => {
        if (conversationActive && !busy && (!recorder || recorder.state !== "recording")) {
          startListening().catch((error) => showError(error, "Microphone error"));
        }
      }, 250);
    }
  }
}

async function loadConversationList() {
  if (!conversationListEl) return;
  try {
    const response = await fetch("/api/conversations");
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    const conversations = Array.isArray(data.conversations) ? data.conversations : [];
    renderConversationList(conversations);
  } catch (error) {
    conversationListEl.innerHTML = "";
    const empty = document.createElement("p");
    empty.className = "conversation-list-empty";
    empty.textContent = "Database unavailable";
    conversationListEl.append(empty);
  }
}

function renderConversationList(conversations) {
  conversationListEl.innerHTML = "";
  if (conversations.length === 0) {
    const empty = document.createElement("p");
    empty.className = "conversation-list-empty";
    empty.textContent = "No saved sessions";
    conversationListEl.append(empty);
    return;
  }
  for (const item of conversations) {
    const button = document.createElement("button");
    button.className = "conversation-list-item";
    if (item.session_id === sessionId) button.classList.add("conversation-list-item-active");
    button.type = "button";
    button.innerHTML = "<strong></strong><span></span><small></small>";
    button.querySelector("strong").textContent = item.title || "Untitled";
    button.querySelector("span").textContent = item.last_content || "";
    button.querySelector("small").textContent = `${item.turn_count || 0} turns`;
    button.addEventListener("click", () => loadConversation(item.session_id));
    conversationListEl.append(button);
  }
}

async function loadConversation(id) {
  if (!id || busy) return;
  conversationActive = false;
  stopBargeInMonitor();
  ttsAbort?.abort();
  window.speechSynthesis?.cancel();
  if (playback) playback.pause();
  if (recorder && recorder.state === "recording") stopRecording("load");
  stopVoiceStream();
  stopTracks();
  setStatus("Loading conversation", "processing");
  const response = await fetch(`/api/conversations/${encodeURIComponent(id)}/load`, { method: "POST" });
  if (!response.ok) {
    showError(new Error(await response.text()));
    return;
  }
  const data = await response.json();
  sessionId = data.session_id;
  resetCurrentConversation();
  for (const turn of data.turns || []) {
    addTurn(turn.role === "assistant" ? "Companion" : turn.role === "user" ? "You" : "System", turn.content);
  }
  await loadConversationList();
  setStatus("Conversation loaded", "ready");
}

async function playAudioChunk(payload) {
  const audio = payload.audio || "";
  if (!audio) return;
  const blob = base64ToBlob(audio, payload.mime_type || "audio/wav");
  if (playback) {
    playback.pause();
    URL.revokeObjectURL(playback.src);
  }
  playback = speakerOutput;
  playback.src = URL.createObjectURL(blob);
  playback.currentTime = 0;
  playback.playbackRate = 1.03;
  setStatus("Speaking", "speaking");
  await startBargeInMonitor().catch(() => {});
  try {
    await new Promise((resolve, reject) => {
      playback.onended = resolve;
      playback.onerror = () => reject(new Error("The browser could not play a streamed voice chunk."));
      playback.play().catch((error) => reject(new Error(`Audio playback failed: ${error.message}`)));
    });
  } finally {
    URL.revokeObjectURL(playback.src);
    playback.removeAttribute("src");
    playback.load();
    stopBargeInMonitor();
  }
}

async function playTts(text) {
  ttsAbort?.abort();
  ttsAbort = new AbortController();
  setStatus("Generating voice", "processing");
  const response = await fetch("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, text }),
    signal: ttsAbort.signal,
  });
  if (!response.ok) {
    const detail = await response.text();
    await speakWithBrowserVoice(text);
    addTurn("System", `Local TTS failed, used browser voice. ${detail}`);
    ttsAbort = null;
    return;
  }
  const blob = await response.blob();
  if (blob.size === 0 || blob.type === "text/plain") {
    await speakWithBrowserVoice(text);
    ttsAbort = null;
    return;
  }
  if (playback) {
    playback.pause();
    URL.revokeObjectURL(playback.src);
  }
  playback = speakerOutput;
  playback.src = URL.createObjectURL(blob);
  playback.currentTime = 0;
  playback.playbackRate = 1.03;
  setStatus("Speaking", "speaking");
  startBargeInMonitor().catch(() => {});

  try {
    await new Promise((resolve, reject) => {
      playback.onended = resolve;
      playback.onerror = () => reject(new Error("The browser could not play the generated voice audio."));
      playback.onpause = () => {
        if (ttsAbort?.signal.aborted) resolve();
      };
      playback.play().catch((error) => {
        reject(
          new Error(
            `Audio playback was blocked or failed: ${error.message}. Tap Mic again to unlock audio.`
          )
        );
      });
    });
  } finally {
    URL.revokeObjectURL(playback.src);
    playback.removeAttribute("src");
    playback.load();
    stopBargeInMonitor();
    ttsAbort = null;
  }
}

async function speakWithBrowserVoice(text) {
  if (!("speechSynthesis" in window)) {
    throw new Error("Local TTS failed and this browser has no speech synthesis fallback.");
  }
  setStatus("Speaking with browser voice", "speaking");
  window.speechSynthesis.cancel();
  startBargeInMonitor().catch(() => {});
  try {
    await new Promise((resolve, reject) => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "en-US";
      utterance.rate = 1.15;
      utterance.pitch = 1.0;
      utterance.onend = resolve;
      utterance.onerror = (event) => reject(new Error(`Browser voice failed: ${event.error}`));
      window.speechSynthesis.speak(utterance);
    });
  } finally {
    stopBargeInMonitor();
  }
}

micButton.addEventListener("click", async () => {
  if (busy && currentPhase === "speaking") {
    await interruptAndListen();
    return;
  }
  if (busy) return;
  if (recorder && recorder.state === "recording") {
    stopRecording("manual");
    micIcon.textContent = "Mic";
    setStatus("Processing voice", "processing");
    return;
  }

  try {
    conversationActive = true;
    await startListening();
  } catch (error) {
    showError(error, "Microphone error");
  }
});

async function startListening() {
  if (busy || (recorder && recorder.state === "recording")) {
    return;
  }
  activeStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recorder = new MediaRecorder(activeStream);
  const chunks = [];
  await startVoiceStream(recorder.mimeType || "audio/webm").catch(() => {
    streamedTranscript = null;
    stopVoiceStream(false);
  });
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size > 0) chunks.push(event.data);
    if (event.data.size > 0) sendVoiceChunk(event.data);
  });
  recorder.addEventListener("stop", async () => {
    stopSilenceMonitor();
    stopTracks();
    await handleRecording(chunks);
  });
  recorder.start(250);
  startSilenceMonitor(activeStream);
  micIcon.textContent = "Listening";
  micButton.title = "Send now";
  setStatus("Listening. Pause to send.", "listening");
}

async function startBargeInMonitor() {
  if (!conversationActive || bargeInMonitor || recorder?.state === "recording") {
    return;
  }
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    return;
  }
  bargeInTriggered = false;
  bargeInStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  bargeInAudioContext = new AudioContextClass();
  const source = bargeInAudioContext.createMediaStreamSource(bargeInStream);
  const analyser = bargeInAudioContext.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);

  const samples = new Float32Array(analyser.fftSize);
  let hotFrames = 0;

  function tick() {
    if (!bargeInAudioContext || currentPhase !== "speaking" || !conversationActive) {
      stopBargeInMonitor();
      return;
    }
    analyser.getFloatTimeDomainData(samples);
    const rms = Math.sqrt(samples.reduce((total, sample) => total + sample * sample, 0) / samples.length);
    hotFrames = rms > bargeInThreshold ? hotFrames + 1 : 0;
    if (hotFrames >= bargeInFramesNeeded && !bargeInTriggered) {
      bargeInTriggered = true;
      interruptAndListen().catch((error) => showError(error, "Barge-in error"));
      return;
    }
    bargeInMonitor = requestAnimationFrame(tick);
  }

  bargeInMonitor = requestAnimationFrame(tick);
}

function stopBargeInMonitor() {
  if (bargeInMonitor) {
    cancelAnimationFrame(bargeInMonitor);
    bargeInMonitor = null;
  }
  if (bargeInAudioContext) {
    bargeInAudioContext.close();
    bargeInAudioContext = null;
  }
  if (bargeInStream) {
    bargeInStream.getTracks().forEach((track) => track.stop());
    bargeInStream = null;
  }
}

async function interruptAndListen() {
  conversationActive = true;
  stopBargeInMonitor();
  ttsAbort?.abort();
  window.speechSynthesis?.cancel();
  if (playback) playback.pause();
  if (sessionId) {
    await fetch(`/api/barge-in/${sessionId}`, { method: "POST" }).catch(() => {});
  }
  setStatus("Interrupted. Listening.", "listening");
  busy = false;
  micButton.disabled = false;
  if (!recorder || recorder.state !== "recording") {
    await startListening();
  }
}

async function startVoiceStream(contentType) {
  stopVoiceStream();
  if (!sessionId) {
    sessionId = crypto.randomUUID();
  }
  streamedTranscript = new Promise((resolve, reject) => {
    streamedTranscriptResolve = resolve;
    streamedTranscriptReject = reject;
  });
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  voiceSocket = new WebSocket(`${protocol}://${window.location.host}/ws/voice/${sessionId}`);
  voiceSocket.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "transcript") {
      streamedTranscriptResolve?.(payload.text || "");
    } else if (payload.type === "error") {
      voiceStreamHandlers.error?.(payload);
      streamedTranscriptReject?.(new Error(payload.message || "Voice stream error"));
      stopVoiceStream(false);
    } else if (voiceStreamHandlers[payload.type]) {
      voiceStreamHandlers[payload.type](payload);
    }
  });
  voiceSocket.addEventListener("error", () => {
    voiceStreamHandlers.error?.({ message: "Voice stream failed." });
    streamedTranscriptReject?.(new Error("Voice stream failed."));
    stopVoiceStream(false);
  });
  await new Promise((resolve, reject) => {
    voiceSocket.addEventListener("open", resolve, { once: true });
    voiceSocket.addEventListener("error", () => reject(new Error("Voice stream failed.")), {
      once: true,
    });
  });
  voiceSocket.send(JSON.stringify({ type: "audio_start", content_type: contentType }));
}

function stopVoiceStream(rejectPending = true) {
  if (rejectPending) {
    streamedTranscriptReject?.(new Error("Voice stream stopped."));
  }
  streamedTranscriptResolve = null;
  streamedTranscriptReject = null;
  if (voiceSocket) {
    voiceSocket.close();
    voiceSocket = null;
  }
}

async function sendVoiceChunk(blob) {
  if (!voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) {
    return;
  }
  const audio = await blobToBase64(blob);
  if (voiceSocket?.readyState === WebSocket.OPEN) {
    voiceSocket.send(JSON.stringify({ type: "audio_chunk", audio }));
  }
}

function endVoiceStream() {
  if (!voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) {
    return false;
  }
  voiceSocket.send(JSON.stringify({ type: "audio_end" }));
  return true;
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error("Could not read audio chunk."));
    reader.readAsDataURL(blob);
  });
}

function base64ToBlob(base64, contentType) {
  const raw = atob(base64);
  const bytes = new Uint8Array(raw.length);
  for (let index = 0; index < raw.length; index += 1) {
    bytes[index] = raw.charCodeAt(index);
  }
  return new Blob([bytes], { type: contentType });
}

function stopRecording(_reason = "auto") {
  if (!recorder || recorder.state !== "recording") {
    return;
  }
  recordingStopReason = _reason;
  recorder.stop();
}

function startSilenceMonitor(stream) {
  stopSilenceMonitor();
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  audioContext = new AudioContextClass();
  const source = audioContext.createMediaStreamSource(stream);
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);

  const samples = new Float32Array(analyser.fftSize);
  recordingStartedAt = performance.now();
  lastVoiceAt = recordingStartedAt;
  speechStarted = false;
  recordingHadSpeech = false;
  recordingStopReason = "manual";

  function tick(now) {
    analyser.getFloatTimeDomainData(samples);
    const rms = Math.sqrt(samples.reduce((total, sample) => total + sample * sample, 0) / samples.length);
    if (rms > silenceThreshold) {
      speechStarted = true;
      recordingHadSpeech = true;
      lastVoiceAt = now;
      setStatus("Listening", "listening");
    } else if (speechStarted) {
      const quietFor = now - lastVoiceAt;
      setStatus(quietFor > 450 ? "Pause detected" : "Listening", "listening");
      if (quietFor >= pauseToSubmitMs) {
        setStatus("Processing voice", "processing");
        stopRecording("silence");
        return;
      }
    } else if (now - recordingStartedAt >= noSpeechTimeoutMs) {
      setStatus("No speech detected", "processing");
      stopRecording("timeout");
      return;
    }

    if (now - recordingStartedAt >= maxRecordingMs) {
      setStatus("Processing voice", "processing");
      stopRecording("max");
      return;
    }
    silenceMonitor = requestAnimationFrame(tick);
  }
  silenceMonitor = requestAnimationFrame(tick);
}

function stopSilenceMonitor() {
  if (silenceMonitor) {
    cancelAnimationFrame(silenceMonitor);
    silenceMonitor = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  speechStarted = false;
}

async function handleRecording(chunks) {
  busy = true;
  micButton.disabled = true;
  let sentToAssistant = false;
  try {
    if (recordingStopReason === "timeout" && !recordingHadSpeech) {
      throw new Error("I did not hear speech. Try again when you are ready.");
    }
    if (chunks.length === 0) throw new Error("No audio was captured.");
    setStatus("Transcribing", "processing");
    let text = "";
    if (endVoiceStream() && streamedTranscript) {
      try {
        text = String(await withTimeout(streamedTranscript, 120000, "Streaming ASR timed out.")).trim();
      } catch {
        text = await transcribeWithRest(chunks);
      }
    } else {
      text = await transcribeWithRest(chunks);
    }
    if (!text) {
      text = await transcribeWithRest(chunks);
    }
    if (!text || text.startsWith("[asr unavailable")) {
      throw new Error(text || "I did not catch that. Try speaking a little longer.");
    }
    sentToAssistant = true;
    await sendText(text);
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    micButton.disabled = false;
    micIcon.textContent = "Mic";
    micButton.title = "Start talking";
    recorder = null;
    stopVoiceStream(false);
    if (conversationActive && !sentToAssistant && !busy) {
      setTimeout(() => {
        if (conversationActive && !busy && (!recorder || recorder.state !== "recording")) {
          startListening().catch((error) => showError(error, "Microphone error"));
        }
      }, 350);
    }
  }
}

function withTimeout(promise, timeoutMs, message) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), timeoutMs);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      }
    );
  });
}

async function transcribeWithRest(chunks) {
  const blob = new Blob(chunks, { type: recorder?.mimeType || "audio/webm" });
  const form = new FormData();
  form.append("audio", blob, "voice.webm");
  if (sessionId) form.append("session_id", sessionId);

  const response = await fetch("/api/asr", { method: "POST", body: form });
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  return (data.text || "").trim();
}

function stopTracks() {
  stopSilenceMonitor();
  if (!activeStream) return;
  activeStream.getTracks().forEach((track) => track.stop());
  activeStream = null;
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function showError(error, prefix = "Error") {
  if (error?.name === "AbortError") {
    setStatus("Interrupted", "ready");
    return;
  }
  const message = error?.message || String(error);
  setStatus(`${prefix}: ${message}`, "error");
  addTurn("System", message);
}

stopButton.addEventListener("click", async () => {
  conversationActive = false;
  stopBargeInMonitor();
  ttsAbort?.abort();
  window.speechSynthesis?.cancel();
  if (playback) playback.pause();
  if (recorder && recorder.state === "recording") stopRecording("barge-in");
  stopVoiceStream();
  stopTracks();
  if (sessionId) await fetch(`/api/barge-in/${sessionId}`, { method: "POST" });
  busy = false;
  micButton.disabled = false;
  micIcon.textContent = "Mic";
  micButton.title = "Start talking";
  setStatus("Interrupted", "ready");
});

newConversationButton.addEventListener("click", async () => {
  conversationActive = false;
  stopBargeInMonitor();
  ttsAbort?.abort();
  window.speechSynthesis?.cancel();
  if (playback) playback.pause();
  if (recorder && recorder.state === "recording") stopRecording("new");
  stopVoiceStream();
  stopTracks();
  sessionId = crypto.randomUUID();
  resetCurrentConversation();
  setStatus("Ready", "ready");
  await loadConversationList();
});

loadConversationList().catch(() => {});
