let sessionId = null;
let recorder = null;
let playback = null;
let activeStream = null;
let ttsAbort = null;
let busy = false;
let turnCount = 0;
let memoryEventCount = 0;

const statusEl = document.querySelector("#status");
const phaseText = document.querySelector("#phaseText");
const conversationEl = document.querySelector("#conversation");
const historyCountEl = document.querySelector("#historyCount");
const memoryActivityEl = document.querySelector("#memoryActivity");
const memoryCountEl = document.querySelector("#memoryCount");
const micButton = document.querySelector("#micButton");
const micIcon = document.querySelector("#micIcon");
const stopButton = document.querySelector("#stopButton");
const voiceStage = document.querySelector(".voice-stage");
const speakerOutput = document.querySelector("#speakerOutput");

function setStatus(text, phase = "ready") {
  statusEl.textContent = text;
  phaseText.textContent = text;
  voiceStage.dataset.phase = phase;
}

function addTurn(role, text, meta = {}) {
  if (!text) return;
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
    setStatus("Ready", "ready");
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    micButton.disabled = false;
    micIcon.textContent = "Mic";
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
  setStatus("Speaking", "speaking");

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
  URL.revokeObjectURL(playback.src);
  playback.removeAttribute("src");
  playback.load();
  ttsAbort = null;
}

async function speakWithBrowserVoice(text) {
  if (!("speechSynthesis" in window)) {
    throw new Error("Local TTS failed and this browser has no speech synthesis fallback.");
  }
  setStatus("Speaking with browser voice", "speaking");
  window.speechSynthesis.cancel();
  await new Promise((resolve, reject) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.rate = 0.98;
    utterance.pitch = 1.0;
    utterance.onend = resolve;
    utterance.onerror = (event) => reject(new Error(`Browser voice failed: ${event.error}`));
    window.speechSynthesis.speak(utterance);
  });
}

micButton.addEventListener("click", async () => {
  if (busy) return;
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    micIcon.textContent = "Mic";
    setStatus("Processing voice", "processing");
    return;
  }

  try {
    activeStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(activeStream);
    const chunks = [];
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    });
    recorder.addEventListener("stop", async () => {
      stopTracks();
      await handleRecording(chunks);
    });
    recorder.start();
    micIcon.textContent = "Done";
    setStatus("Listening", "listening");
  } catch (error) {
    showError(error, "Microphone error");
  }
});

async function handleRecording(chunks) {
  busy = true;
  micButton.disabled = true;
  try {
    if (chunks.length === 0) throw new Error("No audio was captured.");
    const blob = new Blob(chunks, { type: recorder?.mimeType || "audio/webm" });
    const form = new FormData();
    form.append("audio", blob, "voice.webm");
    if (sessionId) form.append("session_id", sessionId);

    setStatus("Transcribing", "processing");
    const response = await fetch("/api/asr", { method: "POST", body: form });
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    const text = (data.text || "").trim();
    if (!text || text.startsWith("[asr unavailable")) {
      throw new Error(text || "I did not catch that. Try speaking a little longer.");
    }
    await sendText(text);
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    micButton.disabled = false;
    micIcon.textContent = "Mic";
    recorder = null;
  }
}

function stopTracks() {
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
  ttsAbort?.abort();
  window.speechSynthesis?.cancel();
  if (playback) playback.pause();
  if (recorder && recorder.state === "recording") recorder.stop();
  stopTracks();
  if (sessionId) await fetch(`/api/barge-in/${sessionId}`, { method: "POST" });
  busy = false;
  micButton.disabled = false;
  micIcon.textContent = "Mic";
  setStatus("Interrupted", "ready");
});
