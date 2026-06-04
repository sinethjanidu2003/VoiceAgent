import { OpenAIRealtimeWS } from "openai/realtime/ws";
import { resample16kTo24k } from "./audioResample.js";

/**
 * Bidirectional bridge to OpenAI Realtime API (public OpenAI key).
 * Input: PCM16 mono 16 kHz or 24 kHz. Output: PCM16 mono 24 kHz deltas.
 */
export async function createRealtimeBridge({
  openai,
  model,
  voice,
  instructions,
  onReady,
  onAudio24,
  onUserTranscript,
  onAssistantTranscript,
  onSpeechStarted,
  onError,
}) {
  const rt = await OpenAIRealtimeWS.create(openai, { model });
  let ready = false;
  /** @type {Buffer[]} */
  const pending24 = [];

  function flushPending() {
    for (const chunk of pending24) sendAppend24(chunk);
    pending24.length = 0;
  }

  function sendAppend24(pcm24) {
    if (!pcm24.length) return;
    rt.send({
      type: "input_audio_buffer.append",
      audio: pcm24.toString("base64"),
    });
  }

  const bridge = {
    appendAudio16k(pcm16) {
      bridge.appendAudio24k(resample16kTo24k(pcm16));
    },

    appendAudio24k(pcm24) {
      if (!ready) {
        pending24.push(pcm24);
        return;
      }
      sendAppend24(pcm24);
    },

    requestGreeting() {
      if (!ready) return;
      rt.send({
        type: "response.create",
        response: { modalities: ["text", "audio"] },
      });
    },

    disconnect() {
      ready = false;
      pending24.length = 0;
      try {
        rt.close();
      } catch {
        /* ignore */
      }
    },
  };

  rt.on("session.created", () => {
    rt.send({
      type: "session.update",
      session: {
        instructions,
        voice,
        modalities: ["text", "audio"],
        input_audio_format: "pcm16",
        output_audio_format: "pcm16",
        input_audio_transcription: { model: "whisper-1" },
        turn_detection: {
          type: "server_vad",
          threshold: 0.5,
          prefix_padding_ms: 300,
          silence_duration_ms: 600,
        },
      },
    });
  });

  rt.on("session.updated", () => {
    if (ready) return;
    ready = true;
    flushPending();
    onReady?.();
  });

  rt.on("response.output_audio.delta", (event) => {
    if (!event.delta) return;
    onAudio24?.(Buffer.from(event.delta, "base64"));
  });

  rt.on("conversation.item.input_audio_transcription.completed", (event) => {
    if (event.transcript) onUserTranscript?.(event.transcript);
  });

  rt.on("response.output_audio_transcript.done", (event) => {
    if (event.transcript) onAssistantTranscript?.(event.transcript);
  });

  rt.on("input_audio_buffer.speech_started", () => {
    onSpeechStarted?.();
  });

  rt.on("error", (err) => {
    onError?.(err);
  });

  return bridge;
}
