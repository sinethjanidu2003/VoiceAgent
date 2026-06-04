/** Browser built-in TTS via Web Speech API (window.speechSynthesis). */

let selectedVoice = null;
/** @type {SpeechSynthesisUtterance[]} */
const queue = [];
let speaking = false;

export function isBrowserTtsSupported() {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

export function loadVoices() {
  return new Promise((resolve) => {
    const voices = speechSynthesis.getVoices();
    if (voices.length) {
      resolve(voices);
      return;
    }
    speechSynthesis.onvoiceschanged = () => resolve(speechSynthesis.getVoices());
    setTimeout(() => resolve(speechSynthesis.getVoices()), 500);
  });
}

export async function initBrowserTts(preferredLang = "en") {
  if (!isBrowserTtsSupported()) return [];
  const voices = await loadVoices();
  selectedVoice =
    voices.find((v) => v.lang.startsWith(preferredLang) && v.localService) ||
    voices.find((v) => v.lang.startsWith(preferredLang)) ||
    voices[0] ||
    null;
  return voices;
}

export function setBrowserVoice(voiceURI) {
  const voice = speechSynthesis.getVoices().find((v) => v.voiceURI === voiceURI);
  if (voice) selectedVoice = voice;
}

export function getSelectedVoice() {
  return selectedVoice;
}

function pumpQueue() {
  if (speaking || !queue.length) return;
  speaking = true;
  const utter = queue.shift();
  utter.onend = () => {
    speaking = false;
    pumpQueue();
  };
  utter.onerror = () => {
    speaking = false;
    pumpQueue();
  };
  speechSynthesis.speak(utter);
}

export function stopBrowserTts() {
  speechSynthesis.cancel();
  queue.length = 0;
  speaking = false;
}

export function speakBrowser(text) {
  if (!isBrowserTtsSupported() || !text?.trim()) return;
  if (speechSynthesis.paused) speechSynthesis.resume();

  const utter = new SpeechSynthesisUtterance(text.trim());
  if (selectedVoice) utter.voice = selectedVoice;
  utter.rate = 1;
  utter.pitch = 1;
  queue.push(utter);
  pumpQueue();
}

/** Speak each sentence as ChatGPT streams text (feels more responsive). */
export function speakBrowserStreamingToken(buffer, delta) {
  const combined = buffer + delta;
  const match = combined.match(/^([\s\S]*?[.!?])([\s\S]*)$/);
  if (!match) return combined;
  const sentence = match[1].trim();
  const rest = match[2];
  if (sentence.length > 3) speakBrowser(sentence);
  return rest;
}

export function flushBrowserStreamBuffer(buffer) {
  const tail = buffer?.trim();
  if (tail) speakBrowser(tail);
  return "";
}
