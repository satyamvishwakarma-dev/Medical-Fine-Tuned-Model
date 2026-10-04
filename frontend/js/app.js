/**
 * BioMistral Clinical Q&A - Frontend Reactive Controller
 * Manages streaming SSE connection, scroll tracking, markdown rendering, and clinical UI states.
 */

// Application State
const state = {
  isStreaming: false,
  abortController: null,
  activeAssistantBubble: null,
  accumulatedResponse: "",
  userScrolledUp: false,
  streamStartTime: null,
  tokenCount: 0,
  settings: {
    temperature: 0.30,
    maxTokens: 512,
    topP: 0.90,
    theme: "dark"
  }
};

// DOM Element Selectors
const elements = {
  chatContainer: document.getElementById("chatContainer"),
  messagesFeed: document.getElementById("messagesFeed"),
  welcomeCard: document.getElementById("welcomeCard"),
  chatForm: document.getElementById("chatForm"),
  userInput: document.getElementById("userInput"),
  sendBtn: document.getElementById("sendBtn"),
  stopBtn: document.getElementById("stopBtn"),
  clearChatBtn: document.getElementById("clearChatBtn"),
  statusBadge: document.getElementById("statusBadge"),
  scrollToBottomBtn: document.getElementById("scrollToBottomBtn"),
  themeToggleBtn: document.getElementById("themeToggleBtn"),
  themeIconDark: document.getElementById("themeIconDark"),
  themeIconLight: document.getElementById("themeIconLight"),
  settingsBtn: document.getElementById("settingsBtn"),
  settingsModal: document.getElementById("settingsModal"),
  closeSettingsBtn: document.getElementById("closeSettingsBtn"),
  saveParamsBtn: document.getElementById("saveParamsBtn"),
  resetParamsBtn: document.getElementById("resetParamsBtn"),
  tempInput: document.getElementById("tempInput"),
  tempVal: document.getElementById("tempVal"),
  maxTokensInput: document.getElementById("maxTokensInput"),
  maxTokensVal: document.getElementById("maxTokensVal"),
  topPInput: document.getElementById("topPInput"),
  topPVal: document.getElementById("topPVal"),
  exportSessionBtn: document.getElementById("exportSessionBtn"),
  disclaimerStrip: document.querySelector(".medical-disclaimer-strip"),
  disclaimerDismiss: document.querySelector(".disclaimer-dismiss")
};

// ==========================================================================
// Initialization & Configuration
// ==========================================================================
function init() {
  loadSavedSettings();
  setupEventListeners();
  checkBackendHealth();
  // Poll health every 20 seconds
  setInterval(checkBackendHealth, 20000);
}

function loadSavedSettings() {
  try {
    const saved = localStorage.getItem("biomistral_settings");
    if (saved) {
      state.settings = { ...state.settings, ...JSON.parse(saved) };
    }
  } catch (e) {
    console.warn("Failed to load settings from storage", e);
  }

  // Apply loaded theme
  applyTheme(state.settings.theme);

  // Sync inputs in settings modal
  elements.tempInput.value = state.settings.temperature;
  elements.tempVal.textContent = Number(state.settings.temperature).toFixed(2);
  elements.maxTokensInput.value = state.settings.maxTokens;
  elements.maxTokensVal.textContent = state.settings.maxTokens;
  elements.topPInput.value = state.settings.topP;
  elements.topPVal.textContent = Number(state.settings.topP).toFixed(2);
}

function saveSettings() {
  state.settings.temperature = parseFloat(elements.tempInput.value);
  state.settings.maxTokens = parseInt(elements.maxTokensInput.value, 10);
  state.settings.topP = parseFloat(elements.topPInput.value);

  try {
    localStorage.setItem("biomistral_settings", JSON.stringify(state.settings));
  } catch (e) {
    console.warn("Failed to save settings", e);
  }
}

// ==========================================================================
// Theme Management
// ==========================================================================
function applyTheme(theme) {
  state.settings.theme = theme;
  document.documentElement.setAttribute("data-theme", theme);
  if (theme === "light") {
    elements.themeIconDark.classList.add("hidden");
    elements.themeIconLight.classList.remove("hidden");
  } else {
    elements.themeIconDark.classList.remove("hidden");
    elements.themeIconLight.classList.add("hidden");
  }
  try {
    localStorage.setItem("biomistral_settings", JSON.stringify(state.settings));
  } catch (e) {}
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  applyTheme(current === "dark" ? "light" : "dark");
}

// ==========================================================================
// Health & Diagnostic Checking
// ==========================================================================
async function checkBackendHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.status === "online" && data.model_available) {
      elements.statusBadge.className = "status-pill online";
      const inMemoryText = data.model_active_in_memory ? " (Warm in VRAM/RAM)" : "";
      elements.statusBadge.querySelector(".status-text").textContent = `BioMistral Ready${inMemoryText}`;
      elements.statusBadge.title = `Model: ${data.target_model} is ready. Ollama active.`;
    } else if (data.status === "online" && !data.model_available) {
      elements.statusBadge.className = "status-pill offline";
      elements.statusBadge.querySelector(".status-text").textContent = "Model Missing";
      elements.statusBadge.title = `Ollama running, but model '${data.target_model}' not found in registry.`;
    } else {
      elements.statusBadge.className = "status-pill offline";
      elements.statusBadge.querySelector(".status-text").textContent = "Ollama Offline";
      elements.statusBadge.title = data.message || "Cannot connect to Ollama.";
    }
  } catch (err) {
    elements.statusBadge.className = "status-pill offline";
    elements.statusBadge.querySelector(".status-text").textContent = "Service Offline";
    elements.statusBadge.title = "Backend server or Ollama is not responding.";
  }
}

// ==========================================================================
// Scroll Management
// ==========================================================================
function scrollToBottom(force = false) {
  if (force || !state.userScrolledUp) {
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
  }
}

function handleContainerScroll() {
  const threshold = 120;
  const distanceFromBottom = elements.chatContainer.scrollHeight - elements.chatContainer.scrollTop - elements.chatContainer.clientHeight;
  
  if (distanceFromBottom > threshold) {
    state.userScrolledUp = true;
    elements.scrollToBottomBtn.classList.remove("hidden");
  } else {
    state.userScrolledUp = false;
    elements.scrollToBottomBtn.classList.add("hidden");
  }
}

// ==========================================================================
// Message Handling & Streaming Execution
// ==========================================================================
async function handleSubmit(promptText) {
  const prompt = (promptText || elements.userInput.value).trim();
  if (!prompt || state.isStreaming) return;

  // Clear input box and reset height
  elements.userInput.value = "";
  elements.userInput.style.height = "auto";

  // Hide welcome triage card once consultation starts
  if (elements.welcomeCard) {
    elements.welcomeCard.style.display = "none";
  }

  // 1. Render User Message Bubble
  renderUserMessage(prompt);
  scrollToBottom(true);

  // 2. Prepare Assistant Bubble with placeholder & typing cursor
  const { card, contentEl, metricsEl } = createAssistantCard();
  elements.messagesFeed.appendChild(card);
  scrollToBottom(true);

  // 3. Initiate Streaming SSE Request
  setStreamingState(true);
  state.accumulatedResponse = "";
  state.tokenCount = 0;
  state.streamStartTime = performance.now();
  state.userScrolledUp = false;

  state.abortController = new AbortController();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        temperature: state.settings.temperature,
        max_tokens: state.settings.maxTokens,
        top_p: state.settings.topP
      }),
      signal: state.abortController.signal
    });

    if (!response.ok) {
      throw new Error(`Server returned error code: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // Keep partial line in buffer

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const raw = line.slice(6).trim();
          if (!raw) continue;

          try {
            const data = JSON.parse(raw);

            if (data.error) {
              contentEl.innerHTML = `<div class="error-msg" style="color: var(--status-error)">⚠️ ${escapeHtml(data.error)}</div>`;
              return;
            }

            if (data.token) {
              state.accumulatedResponse += data.token;
              state.tokenCount++;
              updateAssistantResponse(contentEl, state.accumulatedResponse, true);
              scrollToBottom();
            }

            if (data.done) {
              const elapsedSec = ((performance.now() - state.streamStartTime) / 1000).toFixed(1);
              const speed = (state.tokenCount / (elapsedSec || 1)).toFixed(1);
              metricsEl.textContent = `⚡ Completed in ${elapsedSec}s • ${state.tokenCount} tokens (~${speed} tok/s)`;
              updateAssistantResponse(contentEl, state.accumulatedResponse, false);
              scrollToBottom();
              break;
            }
          } catch (jsonErr) {
            console.warn("SSE JSON parse issue:", jsonErr, line);
          }
        }
      }
    }

  } catch (err) {
    if (err.name === "AbortError") {
      updateAssistantResponse(contentEl, state.accumulatedResponse + "\n\n*(Generation halted by user)*", false);
      metricsEl.textContent = "🛑 Generation halted";
    } else {
      contentEl.innerHTML = `<div class="error-msg" style="color: var(--status-error)">⚠️ Connection failure: ${escapeHtml(err.message)}</div>`;
      metricsEl.textContent = "❌ Error encountered";
    }
  } finally {
    setStreamingState(false);
    scrollToBottom();
  }
}

function setStreamingState(isStreaming) {
  state.isStreaming = isStreaming;
  if (isStreaming) {
    elements.sendBtn.disabled = true;
    elements.stopBtn.classList.remove("hidden");
  } else {
    elements.sendBtn.disabled = false;
    elements.stopBtn.classList.add("hidden");
    state.abortController = null;
  }
}

// ==========================================================================
// DOM Bubble Builders
// ==========================================================================
function renderUserMessage(text) {
  const card = document.createElement("div");
  card.className = "message-card user";
  
  const senderTag = document.createElement("div");
  senderTag.className = "msg-sender-tag";
  senderTag.innerHTML = `<span>Clinical Query</span><span>•</span><span>${formatCurrentTime()}</span>`;

  const wrapper = document.createElement("div");
  wrapper.className = "msg-content-wrapper";
  wrapper.textContent = text;

  card.appendChild(senderTag);
  card.appendChild(wrapper);
  elements.messagesFeed.appendChild(card);
}

function createAssistantCard() {
  const card = document.createElement("div");
  card.className = "message-card assistant";

  // Header with BioMistral identity and current time
  const header = document.createElement("div");
  header.className = "msg-header";
  header.innerHTML = `
    <div class="assistant-identity">
      <div class="assistant-avatar">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 10.5h-5.5V5c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v5.5H5c-.83 0-1.5.67-1.5 1.5s.67 1.5 1.5 1.5h5.5V19c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5v-5.5H19c.83 0 1.5-.67 1.5-1.5s-.67-1.5-1.5-1.5z"/>
        </svg>
      </div>
      <span class="assistant-name">BioMistral Clinical Assistant</span>
      <span class="assistant-badge">Fine-Tuned</span>
    </div>
    <span class="assistant-timestamp">${formatCurrentTime()}</span>
  `;

  // Content Container
  const wrapper = document.createElement("div");
  wrapper.className = "msg-content-wrapper";

  const contentEl = document.createElement("div");
  contentEl.className = "prose";
  contentEl.innerHTML = `<span class="typing-cursor"></span>`;
  wrapper.appendChild(contentEl);

  // Footer with metrics and copy action
  const footer = document.createElement("div");
  footer.className = "msg-action-footer";

  const metricsEl = document.createElement("span");
  metricsEl.className = "metrics-tag";
  metricsEl.textContent = "⚡ Formulating clinical response...";

  const cardActions = document.createElement("div");
  cardActions.className = "card-actions";

  const copyBtn = document.createElement("button");
  copyBtn.className = "mini-action-btn";
  copyBtn.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
    </svg>
    <span>Copy</span>
  `;
  copyBtn.addEventListener("click", () => copyCardText(contentEl, copyBtn));

  cardActions.appendChild(copyBtn);
  footer.appendChild(metricsEl);
  footer.appendChild(cardActions);
  wrapper.appendChild(footer);

  card.appendChild(header);
  card.appendChild(wrapper);

  return { card, contentEl, metricsEl };
}

function updateAssistantResponse(containerEl, markdownText, isGenerating) {
  let parsedHtml = "";
  if (typeof marked !== "undefined" && marked.parse) {
    parsedHtml = marked.parse(markdownText);
  } else {
    parsedHtml = escapeHtml(markdownText).replace(/\n/g, "<br>");
  }

  if (isGenerating) {
    containerEl.innerHTML = parsedHtml + `<span class="typing-cursor"></span>`;
  } else {
    containerEl.innerHTML = parsedHtml;
  }
}

function copyCardText(contentEl, btn) {
  const text = contentEl.innerText || contentEl.textContent;
  navigator.clipboard.writeText(text).then(() => {
    const originalText = btn.innerHTML;
    btn.innerHTML = `<span>✓ Copied</span>`;
    setTimeout(() => {
      btn.innerHTML = originalText;
    }, 1800);
  });
}

// ==========================================================================
// Session Export
// ==========================================================================
function exportConsultationTranscript() {
  const messages = elements.messagesFeed.querySelectorAll(".message-card");
  if (messages.length === 0) {
    alert("No consultation history to export yet.");
    return;
  }

  let transcript = `# BioMistral Clinical Consultation Report\n`;
  transcript += `*Generated: ${new Date().toLocaleString()}*\n`;
  transcript += `*Disclaimer: Educational research artifact only. Not medical advice.*\n\n---\n\n`;

  messages.forEach(card => {
    if (card.classList.contains("user")) {
      const userText = card.querySelector(".msg-content-wrapper").innerText;
      transcript += `### Clinical Question / Query\n${userText}\n\n`;
    } else if (card.classList.contains("assistant")) {
      const botText = card.querySelector(".prose").innerText;
      transcript += `### BioMistral Medical Response\n${botText}\n\n---\n\n`;
    }
  });

  const blob = new Blob([transcript], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `BioMistral_Consultation_${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(url);
}

// ==========================================================================
// Helpers & Utility Functions
// ==========================================================================
function formatCurrentTime() {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(string) {
  return String(string).replace(/[&<>"']/g, function(s) {
    return {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    }[s];
  });
}

// ==========================================================================
// Event Listeners Setup
// ==========================================================================
function setupEventListeners() {
  // Chat Submit
  elements.chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleSubmit();
  });

  // Keyboard shortcut: Enter to submit, Shift+Enter for new line
  elements.userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  });

  // Auto-grow textarea
  elements.userInput.addEventListener("input", function() {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 140) + "px";
  });

  // Stop Generation button
  elements.stopBtn.addEventListener("click", () => {
    if (state.abortController) {
      state.abortController.abort();
    }
  });

  // Scroll to bottom button
  elements.scrollToBottomBtn.addEventListener("click", () => {
    scrollToBottom(true);
  });

  // Track user scroll on chat container
  elements.chatContainer.addEventListener("scroll", handleContainerScroll);

  // Quick Prompt Chips
  document.querySelectorAll(".prompt-chips .chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt");
      if (prompt) {
        handleSubmit(prompt);
      }
    });
  });

  // Clear / New Session
  elements.clearChatBtn.addEventListener("click", () => {
    if (state.isStreaming) return;
    elements.messagesFeed.innerHTML = "";
    if (elements.welcomeCard) {
      elements.welcomeCard.style.display = "block";
    }
    scrollToBottom(true);
  });

  // Theme Toggle
  elements.themeToggleBtn.addEventListener("click", toggleTheme);

  // Settings Modal Open/Close
  elements.settingsBtn.addEventListener("click", () => {
    elements.settingsModal.classList.remove("hidden");
  });
  elements.closeSettingsBtn.addEventListener("click", () => {
    elements.settingsModal.classList.add("hidden");
  });
  elements.settingsModal.addEventListener("click", (e) => {
    if (e.target === elements.settingsModal) {
      elements.settingsModal.classList.add("hidden");
    }
  });

  // Sliders value feedback
  elements.tempInput.addEventListener("input", (e) => {
    elements.tempVal.textContent = Number(e.target.value).toFixed(2);
  });
  elements.maxTokensInput.addEventListener("input", (e) => {
    elements.maxTokensVal.textContent = e.target.value;
  });
  elements.topPInput.addEventListener("input", (e) => {
    elements.topPVal.textContent = Number(e.target.value).toFixed(2);
  });

  // Settings Actions
  elements.saveParamsBtn.addEventListener("click", () => {
    saveSettings();
    elements.settingsModal.classList.add("hidden");
  });

  elements.resetParamsBtn.addEventListener("click", () => {
    elements.tempInput.value = 0.30;
    elements.tempVal.textContent = "0.30";
    elements.maxTokensInput.value = 512;
    elements.maxTokensVal.textContent = "512";
    elements.topPInput.value = 0.90;
    elements.topPVal.textContent = "0.90";
    saveSettings();
  });

  // Export Transcript
  elements.exportSessionBtn.addEventListener("click", exportConsultationTranscript);

  // Dismiss Disclaimer
  if (elements.disclaimerDismiss) {
    elements.disclaimerDismiss.addEventListener("click", () => {
      elements.disclaimerStrip.style.display = "none";
    });
  }
}

// Start on DOM ready
document.addEventListener("DOMContentLoaded", init);
