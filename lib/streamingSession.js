import { toFile } from "openai";
import { pcm16ToWav } from "./wav.js";

const DEFAULT_SILENCE_MS = 1500;

/**
 * Streaming voice session: PCM in → Whisper → ChatGPT stream → text out.
 * Browser clients speak replies with Web Speech API; phone uses OpenAI TTS callback.
 */
export function createStreamingSession({
  openai,
  chatModel,
  systemPrompt,
  sampleRate = 16000,
  silenceMs = DEFAULT_SILENCE_MS,
  minPcmBytes = sampleRate * 2 * 0.3,
  greetingText,
  onReady,
  onUserTranscript,
  onAssistantToken,
  onAssistantText,
  onSpeak,
  onSpeakSentence,
  onInterrupt,
  onError,
  /** 'browser' = sentence chunks for Web Speech API; 'openai' = full reply for phone TTS */
  speakMode = "openai",
}) {
  const messages = [{ role: "system", content: systemPrompt }];
  let audioChunks = [];
  let hadSpeech = false;
  let lastSpeechAt = 0;
  let isProcessing = false;
  let closed = false;
  let greeted = false;
  /** @type {AbortController | null} */
  let activeReply = null;

  function disconnect() {
    closed = true;
    activeReply?.abort();
    activeReply = null;
    audioChunks = [];
  }

  function appendAudio16k(pcm16) {
    if (closed) return;

    if (isProcessing) {
      onInterrupt?.();
      activeReply?.abort();
      activeReply = null;
      isProcessing = false;
      audioChunks = [pcm16];
      hadSpeech = true;
      lastSpeechAt = Date.now();
      return;
    }

    audioChunks.push(pcm16);
    hadSpeech = true;
    lastSpeechAt = Date.now();
    maybeFlush();
  }

  function maybeFlush() {
    if (closed || isProcessing || !hadSpeech || !audioChunks.length) return;
    if (Date.now() - lastSpeechAt < silenceMs) return;

    processUtterance().catch((err) => onError?.(err));
  }

  async function transcribePcm(pcmBuffer) {
    const wav = pcm16ToWav(pcmBuffer, sampleRate);
    const result = await openai.audio.transcriptions.create({
      file: await toFile(wav, "speech.wav", { type: "audio/wav" }),
      model: "whisper-1",
    });
    return result.text?.trim() || "";
  }

  async function streamReply(userText) {
    messages.push({ role: "user", content: userText });
    const controller = new AbortController();
    activeReply = controller;

    const stream = await openai.chat.completions.create(
      {
        model: chatModel,
        messages,
        stream: true,
        max_tokens: 180,
        temperature: 0.7,
      },
      { signal: controller.signal }
    );

    let full = "";
    let sentenceBuf = "";
    for await (const chunk of stream) {
      if (controller.signal.aborted) return null;
      const delta = chunk.choices[0]?.delta?.content || "";
      if (!delta) continue;
      full += delta;
      onAssistantToken?.(delta);

      if (speakMode === "browser" && onSpeakSentence) {
        sentenceBuf += delta;
        const parts = sentenceBuf.split(/(?<=[.!?])\s+/);
        while (parts.length > 1) {
          const sentence = parts.shift()?.trim();
          if (sentence) onSpeakSentence(sentence);
        }
        sentenceBuf = parts[0] || "";
      }
    }

    if (controller.signal.aborted || !full.trim()) return null;

    const reply = full.trim();
    messages.push({ role: "assistant", content: reply });

    if (speakMode === "browser" && onSpeakSentence) {
      const tail = sentenceBuf.trim();
      if (tail) onSpeakSentence(tail);
    }

    onAssistantText?.(reply);

    if (speakMode !== "browser") {
      onSpeak?.(reply);
    }
    return reply;
  }

  async function processUtterance() {
    if (isProcessing || closed) return;

    const pcm = Buffer.concat(audioChunks);
    audioChunks = [];
    hadSpeech = false;

    if (pcm.length < minPcmBytes) return;

    isProcessing = true;
    try {
      const text = await transcribePcm(pcm);
      if (closed) return;
      if (!text) return;

      onUserTranscript?.(text);
      await streamReply(text);
    } catch (err) {
      if (err.name !== "AbortError") onError?.(err);
    } finally {
      isProcessing = false;
      activeReply = null;
    }
  }

  function requestGreeting() {
    if (greeted || closed) return;
    greeted = true;
    const text = greetingText || "Hello! How can I help you today?";
    messages.push({ role: "assistant", content: text });
    onAssistantText?.(text);
    if (speakMode === "browser" && onSpeakSentence) {
      onSpeakSentence(text);
    } else {
      onSpeak?.(text);
    }
  }

  const api = {
    appendAudio16k,
    requestGreeting,
    disconnect,
  };

  queueMicrotask(() => onReady?.());

  return api;
}
