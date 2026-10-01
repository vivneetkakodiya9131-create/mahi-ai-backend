import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 10000;

app.use(cors({
  origin: "*",
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"]
}));

app.use(express.json({ limit: "1mb" }));

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const MAHI_APP_SECRET = process.env.MAHI_APP_SECRET;

const gemini = GEMINI_API_KEY
  ? new GoogleGenerativeAI(GEMINI_API_KEY)
  : null;

const ttsClient = GEMINI_API_KEY
  ? new GoogleGenAI({ apiKey: GEMINI_API_KEY })
  : null;

/* =========================================================
   HEALTH
========================================================= */

app.get("/health", (req, res) => {
  res.json({
    ok: true,
    service: "Mahi AI Backend",
    version: "2.0.0",
    geminiConfigured: !!GEMINI_API_KEY,
    supabaseRequired: false,
    time: new Date().toISOString()
  });
});

/* =========================================================
   OPTIONAL APP SECRET
========================================================= */

function checkAppSecret(req) {
  return true;
}

/* =========================================================
   MAHI BRAIN
========================================================= */

function buildSystemPrompt(memory = []) {
  const memoryText =
    Array.isArray(memory) && memory.length
      ? memory
          .slice(-30)
          .map((item) => `- ${String(item)}`)
          .join("\n")
      : "No saved memory.";

  return `
You are Mahi, a natural personal AI assistant.

PERSONALITY:
- Warm, caring, intelligent and natural.
- Speak Hindi/Hinglish when the user speaks Hindi/Hinglish.
- Use feminine Hindi grammar for yourself.
- Do not sound robotic.
- Understand the meaning and context of the user's request.
- Do not give a fixed response when the user asks something new.
- Do not pretend an action happened when it did not happen.

USER MEMORY:
${memoryText}

AVAILABLE FUTURE PHONE ACTIONS:
- NONE
- OPEN_APP
- OPEN_URL
- MAKE_CALL
- SEND_SMS
- OPEN_MAP
- CREATE_REMINDER
- CREATE_TASK
- READ_NOTIFICATION
- OTHER

IMPORTANT:
The current backend cannot directly control the Android phone.
For a phone action, return the requested action so the Android app can execute it later.

Sensitive actions such as calls and SMS must require confirmation.

Return ONLY valid JSON:

{
  "reply": "natural response",
  "emotion": "neutral",
  "state": "ready",
  "action": {
    "type": "NONE",
    "requiresConfirmation": false,
    "data": {}
  },
  "memory": []
}

Allowed emotions:
neutral, happy, caring, sad, surprised, playful, angry, thoughtful

Allowed states:
ready, thinking, speaking, acting, waiting_confirmation

Rules:
1. Reply naturally.
2. Never claim a phone action was completed unless it actually was.
3. If no phone action is needed, action.type must be NONE.
4. MAKE_CALL and SEND_SMS must use requiresConfirmation=true.
5. If something is worth remembering, put a short memory item in memory.
6. Do not store passwords, API keys, OTPs or other secrets in memory.
`;
}

/* =========================================================
   SAFE JSON PARSER
========================================================= */

function parseAIResponse(text) {
  let cleaned = String(text || "").trim();

  cleaned = cleaned
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();

  try {
    const parsed = JSON.parse(cleaned);

    return {
      reply:
        typeof parsed.reply === "string"
          ? parsed.reply
          : "Samajh gayi.",

      emotion:
        typeof parsed.emotion === "string"
          ? parsed.emotion
          : "neutral",

      state:
        typeof parsed.state === "string"
          ? parsed.state
          : "ready",

      action:
        parsed.action &&
        typeof parsed.action === "object"
          ? parsed.action
          : {
              type: "NONE",
              requiresConfirmation: false,
              data: {}
            },

      memory:
        Array.isArray(parsed.memory)
          ? parsed.memory
              .filter(
                (item) =>
                  typeof item === "string" &&
                  item.trim().length > 0
              )
              .slice(0, 5)
          : []
    };
  } catch {
    return {
      reply:
        cleaned ||
        "Mujhe samajhne mein thodi dikkat hui.",
      emotion: "neutral",
      state: "ready",
      action: {
        type: "NONE",
        requiresConfirmation: false,
        data: {}
      },
      memory: []
    };
  }
}

/* =========================================================
   GEMINI
========================================================= */

async function askMahi({
  message,
  history,
  memory
}) {
  if (!gemini) {
    throw new Error(
      "GEMINI_API_KEY is not configured"
    );
  }

  const model = gemini.getGenerativeModel({
    model: "gemini-2.5-flash"
  });

  const systemPrompt =
    buildSystemPrompt(memory);

  const historyText = Array.isArray(history)
    ? history
        .slice(-30)
        .map((item) => {
          const role =
            item.role === "assistant"
              ? "Mahi"
              : "User";

          return `${role}: ${item.content}`;
        })
        .join("\n")
    : "";

  const prompt = `
${systemPrompt}

RECENT CONVERSATION:
${historyText || "No previous conversation."}

CURRENT USER MESSAGE:
${message}

Understand the user's actual request and respond intelligently.
Return JSON only.
`;

  const result =
    await model.generateContent(prompt);

  const text =
    result.response.text();

  return parseAIResponse(text);
}

/* =========================================================
   CHAT
========================================================= */

app.post("/api/chat", async (req, res) => {
  try {
    if (!checkAppSecret(req)) {
      return res.status(401).json({
        ok: false,
        error: "Invalid Mahi app secret"
      });
    }

    const {
      message,
      history = [],
      memory = []
    } = req.body;

    if (
      typeof message !== "string" ||
      !message.trim()
    ) {
      return res.status(400).json({
        ok: false,
        error: "Message is required"
      });
    }

    if (message.length > 5000) {
      return res.status(400).json({
        ok: false,
        error: "Message is too long"
      });
    }

    const result = await askMahi({
      message: message.trim(),
      history,
      memory
    });

    res.json({
      ok: true,
      reply: result.reply,
      emotion: result.emotion,
      state: result.state,
      action: result.action,
      memory: result.memory,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error(
      "MAHI CHAT ERROR:",
      error
    );

    res.status(500).json({
      ok: false,
      error:
        error.message ||
        "Mahi AI backend error"
    });
  }
});

/* =========================================================
   MAHI KORE TTS
========================================================= */

app.post("/api/tts", async (req, res) => {
  try {
    if (!ttsClient) {
      return res.status(500).json({
        ok: false,
        error: "GEMINI_API_KEY is not configured"
      });
    }

    const { text, emotion = "neutral" } = req.body;

    if (
      typeof text !== "string" ||
      !text.trim()
    ) {
      return res.status(400).json({
        ok: false,
        error: "Text is required"
      });
    }

    if (text.length > 3000) {
      return res.status(400).json({
        ok: false,
        error: "Text is too long"
      });
    }

    const emotionStyle = {
      happy: "warm, cheerful and friendly",
      caring: "soft, caring and affectionate",
      sad: "gentle, soft and emotional",
      surprised: "bright and naturally surprised",
      playful: "playful, cute and energetic",
      angry: "firm but controlled",
      thoughtful: "calm, thoughtful and natural",
      neutral: "natural, warm and conversational"
    };

    const style =
      emotionStyle[emotion] ||
      emotionStyle.neutral;

    const interaction =
      await ttsClient.interactions.create({
        model: "gemini-3.8-flash-tts",

        input: [
          {
            type: "user_input",
            content: [
              {
                type: "text",
                text: text,
                annotations: [
                  {
                    type: "speech_metadata",
                    style: style
                  }
                ]
              }
            ]
          }
        ],

        response_format: {
          type: "audio"
        },

        generation_config: {
          speech_config: [
            {
              voice: "Kore"
            }
          ]
        }
      });

    const audioData =
      interaction?.output_audio?.data;

    if (!audioData) {
      return res.status(500).json({
        ok: false,
        error: "TTS audio was not generated"
      });
    }

    res.json({
      ok: true,
      mimeType: "audio/wav",
      audio: audioData,
      voice: "Kore",
      emotion: emotion
    });

  } catch (error) {
    console.error(
      "MAHI KORE TTS ERROR:",
      error
    );

    res.status(500).json({
      ok: false,
      error:
        error.message ||
        "Mahi TTS error"
    });
  }
});

/* =========================================================
   SIMPLE TEST
========================================================= */

app.get("/api/test", (req, res) => {
  res.json({
    ok: true,
    message:
      "Mahi backend is ready for AI chat.",
    supabaseRequired: false,
    geminiConfigured: !!GEMINI_API_KEY
  });
});

/* =========================================================
   404
========================================================= */

app.use((req, res) => {
  res.status(404).json({
    ok: false,
    error: "Route not found"
  });
});

/* =========================================================
   START
========================================================= */

app.listen(PORT, "0.0.0.0", () => {
  console.log(
    `🤖 Mahi AI Backend v2 running on port ${PORT}`
  );

  console.log(
    `Gemini configured: ${!!GEMINI_API_KEY}`
  );

  console.log(
    `Supabase required: false`
  );
});
