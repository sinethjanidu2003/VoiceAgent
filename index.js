import "dotenv/config";
import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import express from "express";
import { WebSocketServer } from "ws";
import { v4 as uuidv4 } from "uuid";
import OpenAI, { toFile } from "openai";
import { CallAutomationClient, StreamingData } from "@azure/communication-call-automation";
import { pcm16ToWav } from "./lib/wav.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const AUDIO_DIR = path.join(__dirname, "tmp", "audio");
fs.mkdirSync(AUDIO_DIR, { recursive: true });

const PORT = process.env.PORT || 3000;
const app = express();
app.use(express.json());
app.use("/audio", express.static(AUDIO_DIR));
app.use(express.static(path.join(__dirname, "public")));

const config = {
  connectionString: process.env.CONNECTION_STRING?.trim(),
  fromPhone: process.env.FROM_PHONE_NUMBER?.trim(),
  callbackUri: process.env.CALLBACK_URI?.trim().replace(/\/$/, ""),
  openAiKey: process.env.OPENAI_API_KEY?.trim(),
  chatModel: process.env.OPENAI_CHAT_MODEL?.trim() || "gpt-4o-mini",
  ttsVoice: process.env.OPENAI_TTS_VOICE?.trim() || "nova",
};

/** @type {CallAutomationClient | null} */
let acsClient = null;

/** @type {OpenAI | null} */
let openai = null;

/** @type {Map<string, object>} */
const callSessions = new Map();

const transcriptLog = [];
const MAX_LOG = 200;
const SAMPLE_RATE = 16000;
const SILENCE_MS = 1500;
const MIN_PCM_BYTES = SAMPLE_RATE * 2 * 0.3; // 0.3 sec minimum

const SYSTEM_PROMPT = `You are a friendly phone appointment scheduling assistant.
Help callers book, reschedule, or cancel appointments.
Keep every reply short (1-3 sentences) because it will be read aloud on a phone call.
Ask one question at a time.
Reply in plain spoken English only. No markdown or lists.`;

const HELLO =
  "Hello! I'm your appointment assistant. How can I help you book or change an appointment today?";
const SILENCE = "I didn't catch that. Please tell me how I can help.";
const GOODBYE = "Thank you for calling. Goodbye!";

function logTranscript(callConnectionId, role, text) {
  transcriptLog.push({
    time: new Date().toISOString(),
    callConnectionId,
    role,
    text,
  });
  if (transcriptLog.length > MAX_LOG) transcriptLog.shift();
  console.log(`[${role}] ${text}`);
}

function normalizePhone(phone) {
  const digits = phone.replace(/\D/g, "");
  if (digits.length === 10) return "+1" + digits;
  if (digits.length === 11 && digits.startsWith("1")) return "+" + digits;
  if (phone.startsWith("+")) return phone;
  return "+" + digits;
}

function validateConfig() {
  const required = [
    ["CONNECTION_STRING", config.connectionString],
    ["FROM_PHONE_NUMBER", config.fromPhone],
    ["CALLBACK_URI", config.callbackUri],
    ["OPENAI_API_KEY", config.openAiKey],
  ];
  const missing = required.filter(([, v]) => !v).map(([k]) => k);
  if (missing.length) {
    console.warn("Missing .env:", missing.join(", "));
    return false;
  }
  return true;
}

function initClients() {
  if (!validateConfig()) return;
  acsClient = new CallAutomationClient(config.connectionString);
  openai = new OpenAI({ apiKey: config.openAiKey });
  console.log("Ready: phone → Whisper → ChatGPT → OpenAI TTS (public API)");
}

function mediaWsUrl() {
  return (
    config.callbackUri.replace(/^https:\/\//i, "wss://").replace(/^http:\/\//i, "ws://") +
    "/media"
  );
}

function mediaStreamingOptions() {
  return {
    transportType: "websocket",
    transportUrl: mediaWsUrl(),
    contentType: "audio",
    audioChannelType: "mixed",
    startMediaStreaming: true,
    enableBidirectional: false,
    enableDtmfTones: false,
    audioFormat: "Pcm16KMono",
  };
}

function getSession(callConnectionId) {
  if (!callSessions.has(callConnectionId)) {
    callSessions.set(callConnectionId, {
      mode: "idle",
      audioChunks: [],
      hadSpeech: false,
      lastSpeechAt: 0,
      isProcessing: false,
      silenceRetries: 2,
      greeted: false,
      messages: [{ role: "system", content: SYSTEM_PROMPT }],
    });
  }
  return callSessions.get(callConnectionId);
}

function getCallMedia(id) {
  return acsClient.getCallConnection(id).getCallMedia();
}

async function hangUp(callConnectionId) {
  callSessions.delete(callConnectionId);
  await acsClient.getCallConnection(callConnectionId).hangUp(true);
}

/** OpenAI TTS → save mp3 → play on phone via ACS file URL */
async function speakOnPhone(callConnectionId, text, operationContext) {
  const session = getSession(callConnectionId);
  session.mode = "playing";

  const speech = await openai.audio.speech.create({
    model: "tts-1",
    voice: config.ttsVoice,
    input: text,
  });

  const fileId = uuidv4();
  const filePath = path.join(AUDIO_DIR, `${fileId}.mp3`);
  const audioUrl = `${config.callbackUri}/audio/${fileId}.mp3`;
  fs.writeFileSync(filePath, Buffer.from(await speech.arrayBuffer()));

  await getCallMedia(callConnectionId).playToAll(
    [{ kind: "fileSource", url: audioUrl }],
    { operationContext }
  );

  scheduleListeningFallback(callConnectionId, text, operationContext);
}

async function transcribePcm(pcmBuffer) {
  const wav = pcm16ToWav(pcmBuffer, SAMPLE_RATE);
  const result = await openai.audio.transcriptions.create({
    file: await toFile(wav, "speech.wav", { type: "audio/wav" }),
    model: "whisper-1",
  });
  return result.text?.trim() || "";
}

async function askChatGpt(session, userText) {
  session.messages.push({ role: "user", content: userText });
  const response = await openai.chat.completions.create({
    model: config.chatModel,
    messages: session.messages,
    max_tokens: 180,
    temperature: 0.7,
  });
  const reply =
    response.choices[0]?.message?.content?.trim() || "Sorry, could you repeat that?";
  session.messages.push({ role: "assistant", content: reply });
  return reply;
}

function wantsGoodbye(text) {
  return /\b(bye|goodbye|that's all|nothing else|no thanks|hang up)\b/i.test(text);
}

function scheduleListeningFallback(callConnectionId, text, operationContext) {
  // If ACS PlayCompleted webhook is delayed/missed, still start listening
  const estimatedMs = Math.min(45000, Math.max(5000, text.length * 70));
  setTimeout(() => {
    const s = callSessions.get(callConnectionId);
    if (!s || s.mode !== "playing") return;
    console.log(
      `[${callConnectionId}] PlayCompleted fallback (${operationContext}) → listening`
    );
    s.mode = "listening";
    s.audioChunks = [];
    s.hadSpeech = false;
  }, estimatedMs);
}

function maybeFlushSpeech(callConnectionId) {
  const session = getSession(callConnectionId);
  if (session.mode !== "listening" || session.isProcessing) return;
  if (
    session.hadSpeech &&
    session.audioChunks.length > 0 &&
    Date.now() - session.lastSpeechAt >= SILENCE_MS
  ) {
    processUserAudio(callConnectionId).catch((err) =>
      console.error("processUserAudio:", err.message)
    );
  }
}

async function processUserAudio(callConnectionId) {
  const session = getSession(callConnectionId);
  if (session.isProcessing || session.mode !== "listening") return;

  const pcm = Buffer.concat(session.audioChunks);
  session.audioChunks = [];
  session.hadSpeech = false;

  if (pcm.length < MIN_PCM_BYTES) {
    console.log(`[${callConnectionId}] Audio too short (${pcm.length} bytes), keep listening`);
    return;
  }

  session.isProcessing = true;
  session.mode = "processing";

  try {
    const text = await transcribePcm(pcm);
    console.log(`[${callConnectionId}] Whisper: "${text || "(empty)"}"`);
    if (!text) {
      session.silenceRetries -= 1;
      if (session.silenceRetries > 0) {
        logTranscript(callConnectionId, "system", "(no speech detected)");
        await speakOnPhone(callConnectionId, SILENCE, "Silence");
      } else {
        await speakOnPhone(callConnectionId, GOODBYE, "Goodbye");
      }
      return;
    }

    session.silenceRetries = 2;
    logTranscript(callConnectionId, "user", text);

    if (wantsGoodbye(text)) {
      logTranscript(callConnectionId, "assistant", GOODBYE);
      await speakOnPhone(callConnectionId, GOODBYE, "Goodbye");
      return;
    }

    const reply = await askChatGpt(session, text);
    logTranscript(callConnectionId, "assistant", reply);
    await speakOnPhone(callConnectionId, reply, "Conversation");
  } catch (err) {
    console.error("Process audio error:", err.message);
    await speakOnPhone(callConnectionId, "Sorry, something went wrong. Please try again.", "Error");
  } finally {
    session.isProcessing = false;
  }
}

function onAudioPacket(callConnectionId, base64, isSilent) {
  const session = getSession(callConnectionId);
  if (session.mode !== "listening" || session.isProcessing) return;

  const buf = Buffer.from(base64, "base64");
  if (!buf.length) return;

  // PSTN streams often mark speech as isSilent — record all audio while listening
  session.audioChunks.push(buf);
  session.lastSpeechAt = Date.now();
  session.hadSpeech = true;

  maybeFlushSpeech(callConnectionId);
}

function onMediaMessage(callConnectionId, packetData) {
  const parsed = StreamingData.parse(packetData);
  const kind = StreamingData.getStreamingKind();
  const session = getSession(callConnectionId);

  if (kind === "AudioMetadata") {
    console.log("Audio stream ready:", callConnectionId);
    if (!session.greeted) {
      session.greeted = true;
      logTranscript(callConnectionId, "system", "Call connected — agent speaking");
      speakOnPhone(callConnectionId, HELLO, "Greeting").catch(console.error);
    }
    return;
  }

  if (kind === "AudioData" && parsed.data) {
    onAudioPacket(callConnectionId, parsed.data, parsed.isSilent);
  }
}

async function onPlayCompleted(callConnectionId, context) {
  const session = getSession(callConnectionId);
  console.log(`[${callConnectionId}] PlayCompleted: ${context}`);

  if (context === "Goodbye") {
    await hangUp(callConnectionId);
    return;
  }

  session.mode = "listening";
  session.audioChunks = [];
  session.hadSpeech = false;
  console.log(`[${callConnectionId}] Now listening for speech…`);
}

function onPlayStarted(callConnectionId, context) {
  console.log(`[${callConnectionId}] PlayStarted: ${context}`);
}

// --- HTTP ---

app.get("/api/health", (_req, res) => {
  res.json({
    status: validateConfig() ? "ok" : "missing_config",
    flow: "Phone → Whisper (OpenAI) → ChatGPT (OpenAI) → TTS (OpenAI) → Phone",
    chatModel: config.chatModel,
  });
});

app.get("/api/transcript", (_req, res) => res.json(transcriptLog));

app.post("/call", async (req, res) => {
  if (!acsClient) {
    return res.status(503).json({ success: false, error: "Fill in .env and restart" });
  }

  try {
    const { toPhone } = req.body;
    if (!toPhone) return res.status(400).json({ success: false, error: "toPhone required" });

    if (!config.fromPhone) {
      return res.status(400).json({
        success: false,
        error: "FROM_PHONE_NUMBER is missing in .env (required for outbound PSTN)",
      });
    }

    const normalized = normalizePhone(toPhone);
    const fromNumber = normalizePhone(config.fromPhone);
    const callbackUrl = `${config.callbackUri}/api/callbacks/${uuidv4()}`;

    // sourceCallIdNumber must be on the CallInvite (1st arg), not in options — SDK quirk
    const result = await acsClient.createCall(
      {
        targetParticipant: { phoneNumber: normalized },
        sourceCallIdNumber: { phoneNumber: fromNumber },
      },
      callbackUrl,
      {
        mediaStreamingOptions: mediaStreamingOptions(),
      }
    );

    const callConnectionId = result.callConnection.callConnectionId;
    logTranscript(callConnectionId, "system", `Calling ${normalized}…`);

    res.json({ success: true, callConnectionId, toPhone: normalized });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

app.post("/api/callbacks/:contextId", async (req, res) => {
  res.sendStatus(200);
  const event = (Array.isArray(req.body) ? req.body : [req.body])[0];
  if (!event?.type) return;

  const eventData = event.data || {};
  const callConnectionId = eventData.callConnectionId;
  console.log("ACS callback:", event.type, callConnectionId, eventData.operationContext || "");

  if (!callConnectionId) return;

  try {
    const type = event.type;
    if (
      type === "Microsoft.Communication.PlayCompleted" ||
      type === "Microsoft.Communication.playCompleted"
    ) {
      await onPlayCompleted(callConnectionId, eventData.operationContext);
    } else if (
      type === "Microsoft.Communication.PlayStarted" ||
      type === "Microsoft.Communication.playStarted"
    ) {
      onPlayStarted(callConnectionId, eventData.operationContext);
    } else if (
      type === "Microsoft.Communication.PlayFailed" ||
      type === "Microsoft.Communication.playFailed"
    ) {
      console.error("Play failed:", eventData.resultInformation);
      const session = getSession(callConnectionId);
      session.mode = "listening";
      session.audioChunks = [];
    } else if (type === "Microsoft.Communication.CallDisconnected") {
      callSessions.delete(callConnectionId);
      logTranscript(callConnectionId, "system", "Call ended");
    }
  } catch (err) {
    console.error("Callback error:", err.message);
  }
});

app.delete("/hangup/:id", async (req, res) => {
  try {
    await hangUp(req.params.id);
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// --- Server + ACS audio WebSocket ---

const server = http.createServer(app);
const mediaWss = new WebSocketServer({ noServer: true });

server.on("upgrade", (req, socket, head) => {
  const { pathname } = new URL(req.url, `http://${req.headers.host}`);
  if (pathname === "/media") {
    mediaWss.handleUpgrade(req, socket, head, (ws) => mediaWss.emit("connection", ws, req));
  } else {
    socket.destroy();
  }
});

mediaWss.on("connection", (ws, req) => {
  const callConnectionId = req.headers["x-ms-call-connection-id"];
  if (!callConnectionId) {
    ws.close();
    return;
  }
  console.log("ACS audio stream:", callConnectionId);
  ws.on("message", (data) => {
    try {
      onMediaMessage(callConnectionId, data);
    } catch (err) {
      console.error("Media error:", err.message);
    }
  });
});

server.listen(PORT, () => {
  console.log(`Phone voice agent → http://localhost:${PORT}`);
  initClients();
});
