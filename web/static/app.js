let sessionId = null;
let recorder = null;
let playback = null;

const statusEl = document.querySelector("#status");
const conversationEl = document.querySelector("#conversation");
const chatForm = document.querySelector("#chatForm");
const chatInput = document.querySelector("#chatInput");
const micButton = document.querySelector("#micButton");
const stopButton = document.querySelector("#stopButton");
const memoryList = document.querySelector("#memoryList");

function setStatus(text) {
  statusEl.textContent = text;
}

function addTurn(role, text) {
  const turn = document.createElement("article");
  turn.className = "turn";
  turn.innerHTML = `<strong>${role}</strong><p></p>`;
  turn.querySelector("p").textContent = text;
  conversationEl.append(turn);
  conversationEl.scrollTop = conversationEl.scrollHeight;
}

async function sendText(text) {
  addTurn("You", text);
  setStatus("Thinking");
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, text }),
  });
  const data = await response.json();
  sessionId = data.session_id;
  addTurn("Companion", data.text);
  if (data.memory_stored) {
    const item = document.createElement("li");
    item.textContent = text;
    memoryList.prepend(item);
  }
  await playTts(data.text);
  setStatus("Ready");
}

async function playTts(text) {
  const response = await fetch("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, text }),
  });
  const blob = await response.blob();
  if (blob.size === 0 || blob.type === "text/plain") return;
  if (playback) playback.pause();
  playback = new Audio(URL.createObjectURL(blob));
  await playback.play().catch(() => {});
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;
  chatInput.value = "";
  await sendText(text);
});

micButton.addEventListener("click", async () => {
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    micButton.textContent = "Mic";
    setStatus("Processing voice");
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recorder = new MediaRecorder(stream);
  const chunks = [];
  recorder.addEventListener("dataavailable", (event) => chunks.push(event.data));
  recorder.addEventListener("stop", async () => {
    stream.getTracks().forEach((track) => track.stop());
    const blob = new Blob(chunks, { type: "audio/webm" });
    const form = new FormData();
    form.append("audio", blob, "voice.webm");
    if (sessionId) form.append("session_id", sessionId);
    const response = await fetch("/api/asr", { method: "POST", body: form });
    const data = await response.json();
    const text = data.text || "";
    if (text) await sendText(text);
    setStatus("Ready");
  });
  recorder.start();
  micButton.textContent = "Done";
  setStatus("Listening");
});

stopButton.addEventListener("click", async () => {
  if (playback) playback.pause();
  if (recorder && recorder.state === "recording") recorder.stop();
  if (sessionId) await fetch(`/api/barge-in/${sessionId}`, { method: "POST" });
  setStatus("Interrupted");
});

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`#${button.dataset.tab}Panel`).classList.add("active");
  });
});

async function loadRegistry() {
  const [prompts, tools] = await Promise.all([
    fetch("/api/prompts").then((response) => response.json()),
    fetch("/api/tools").then((response) => response.json()),
  ]);
  document.querySelector("#promptList").innerHTML = prompts.prompts
    .map((prompt) => `<li><strong>${prompt.name}</strong><br>${prompt.description}</li>`)
    .join("");
  document.querySelector("#toolList").innerHTML = tools.tools
    .map((tool) => `<li><strong>${tool.name}</strong><br>${tool.description}</li>`)
    .join("");
}

document.querySelector("#searchForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = document.querySelector("#searchInput").value.trim();
  if (!query) return;
  setStatus("Searching");
  const response = await fetch("/api/tools/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: "web_search", args: { query } }),
  });
  const data = await response.json();
  addTurn("Tool", JSON.stringify(data.result, null, 2));
  setStatus("Ready");
});

loadRegistry();
