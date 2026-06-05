let sessionId = null;
let recorder = null;
let playback = null;
let activeStream = null;
let busy = false;

const statusEl = document.querySelector("#status");
const phaseText = document.querySelector("#phaseText");
const conversationEl = document.querySelector("#conversation");
const micButton = document.querySelector("#micButton");
const micIcon = document.querySelector("#micIcon");
const stopButton = document.querySelector("#stopButton");

function setStatus(text) {
  statusEl.textContent = text;
  phaseText.textContent = text;
}

function addTurn(role, text) {
  if (!text) return;
  const turn = document.createElement("article");
  turn.className = "turn";
  turn.innerHTML = `<strong>${role}</strong><p></p>`;
  turn.querySelector("p").textContent = text;
  conversationEl.append(turn);
  conversationEl.scrollTop = conversationEl.scrollHeight;
}

async function sendText(text) {
  busy = true;
  micButton.disabled = true;
  addTurn("You", text);
  setStatus("Thinking");
  try {
    const data = await postJson("/api/chat", { session_id: sessionId, text });
    sessionId = data.session_id;
    addTurn("Companion", data.text);
    setStatus("Speaking");
    await playTts(data.text);
    setStatus("Ready");
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    micButton.disabled = false;
    micIcon.textContent = "Mic";
  }
}

async function playTts(text) {
  const response = await fetch("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, text }),
  });
  if (!response.ok) throw new Error(await response.text());
  const blob = await response.blob();
  if (blob.size === 0 || blob.type === "text/plain") return;
  if (playback) playback.pause();
  playback = new Audio(URL.createObjectURL(blob));
  await playback.play().catch(() => {});
}

micButton.addEventListener("click", async () => {
  if (busy) return;
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    micIcon.textContent = "Mic";
    setStatus("Processing voice");
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
    setStatus("Listening");
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

    setStatus("Transcribing");
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
  const message = error?.message || String(error);
  setStatus(`${prefix}: ${message}`);
  addTurn("System", message);
}

stopButton.addEventListener("click", async () => {
  if (playback) playback.pause();
  if (recorder && recorder.state === "recording") recorder.stop();
  stopTracks();
  if (sessionId) await fetch(`/api/barge-in/${sessionId}`, { method: "POST" });
  busy = false;
  micButton.disabled = false;
  micIcon.textContent = "Mic";
  setStatus("Interrupted");
});
