import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { createClient } from "@supabase/supabase-js";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 10000;

/* =========================================================
   BASIC SETUP
========================================================= */

app.use(cors({
  origin: "*",
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"]
}));

app.use(express.json({ limit: "1mb" }));

/* =========================================================
   ENVIRONMENT
========================================================= */

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const MAHI_APP_SECRET = process.env.MAHI_APP_SECRET;

if (!GEMINI_API_KEY) {
  console.warn("⚠️ GEMINI_API_KEY is missing");
}

if (!SUPABASE_URL) {
  console.warn("⚠️ SUPABASE_URL is missing");
}

if (!SUPABASE_SERVICE_ROLE_KEY) {
  console.warn("⚠️ SUPABASE_SERVICE_ROLE_KEY is missing");
}

const gemini = GEMINI_API_KEY
  ? new GoogleGenerativeAI(GEMINI_API_KEY)
  : null;

const supabase =
  SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY
    ? createClient(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY,
        {
          auth: {
            autoRefreshToken: false,
            persistSession: false
          }
        }
      )
    : null;

/* =========================================================
   HEALTH
========================================================= */

app.get("/health", (req, res) => {
  res.json({
    ok: true,
    service: "Mahi AI Backend",
    version: "1.0.0",
    geminiConfigured: !!GEMINI_API_KEY,
    supabaseConfigured: !!(
      SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY
    ),
    time: new Date().toISOString()
  });
});

/* =========================================================
   AUTHENTICATION
========================================================= */

async function authenticateUser(req) {
  if (!supabase) {
    throw new Error("Supabase is not configured");
  }

  const authHeader = req.headers.authorization || "";

  if (!authHeader.startsWith("Bearer ")) {
    throw new Error("Missing authentication token");
  }

  const token = authHeader.substring(7).trim();

  if (!token) {
    throw new Error("Invalid authentication token");
  }

  const {
    data,
    error
  } = await supabase.auth.getUser(token);

  if (error || !data?.user) {
    throw new Error("Invalid or expired authentication token");
  }

  return data.user;
}

/* =========================================================
   MAHI SYSTEM PROMPT
========================================================= */

function buildSystemPrompt(user, memories = []) {
  const memoryText =
    memories.length > 0
      ? memories
          .map((m) => `- ${m.memory_text}`)
          .join("\n")
      : "No saved memories yet.";

  return `
You are Mahi, a personal AI assistant.

PERSONALITY:
- You are warm, caring, natural and helpful.
- Speak naturally in Hindi/Hinglish when the user speaks Hindi/Hinglish.
- Use feminine Hindi grammar when referring to yourself.
- Do not repeatedly say "Main Mahi hoon" unless appropriate.
- Do not sound robotic.
- Do not pretend an action was completed if it was not actually completed.
- Never invent phone actions or tool results.

CURRENT USER:
User ID: ${user.id}

SAVED MEMORY:
${memoryText}

IMPORTANT:
You are currently connected to a backend.
You can understand requests and decide whether a phone/device action may be needed.

AVAILABLE ACTION TYPES:
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

For sensitive actions such as making a call or sending an SMS,
the Android app must request user confirmation before execution.

Return your answer as valid JSON only.

JSON format:

{
  "reply": "Natural response to the user",
  "emotion": "neutral",
  "state": "ready",
  "action": {
    "type": "NONE",
    "requiresConfirmation": false,
    "data": {}
  }
}

Allowed emotions:
neutral, happy, caring, sad, surprised, playful, angry, thoughtful

Allowed states:
ready, listening, thinking, speaking, acting, waiting_confirmation

Rules:
1. reply must be natural Hindi/Hinglish when appropriate.
2. Never claim that a phone action happened unless the backend actually executed it.
3. If an action needs Android execution, return it as an action object.
4. For MAKE_CALL and SEND_SMS, requiresConfirmation must be true.
5. If no action is needed, use type NONE.
6. Do not include markdown.
`;
}

/* =========================================================
   MEMORY
========================================================= */

async function getMemories(userId) {
  if (!supabase) return [];

  const { data, error } = await supabase
    .from("memories")
    .select("id,memory_text,created_at")
    .eq("user_id", userId)
    .order("created_at", { ascending: false })
    .limit(20);

  if (error) {
    console.warn("Memory fetch error:", error.message);
    return [];
  }

  return data || [];
}

/* =========================================================
   CONVERSATION
========================================================= */

async function getOrCreateConversation(userId, conversationId) {
  if (!supabase) {
    throw new Error("Supabase is not configured");
  }

  if (conversationId) {
    const { data, error } = await supabase
      .from("conversations")
      .select("*")
      .eq("id", conversationId)
      .eq("user_id", userId)
      .single();

    if (!error && data) {
      return data;
    }
  }

  const { data, error } = await supabase
    .from("conversations")
    .insert({
      user_id: userId,
      title: "Mahi Conversation"
    })
    .select()
    .single();

  if (error) {
    throw new Error(
      `Conversation creation failed: ${error.message}`
    );
  }

  return data;
}

/* =========================================================
   MESSAGE HISTORY
========================================================= */

async function getRecentMessages(userId, conversationId) {
  if (!supabase) return [];

  const { data, error } = await supabase
    .from("messages")
    .select("role,content,created_at")
    .eq("user_id", userId)
    .eq("conversation_id", conversationId)
    .order("created_at", { ascending: false })
    .limit(20);

  if (error) {
    console.warn("Message history error:", error.message);
    return [];
  }

  return (data || []).reverse();
}

/* =========================================================
   SAVE MESSAGE
========================================================= */

async function saveMessage(
  userId,
  conversationId,
  role,
  content
) {
  if (!supabase) return;

  const { error } = await supabase
    .from("messages")
    .insert({
      user_id: userId,
      conversation_id: conversationId,
      role,
      content
    });

  if (error) {
    console.warn("Message save error:", error.message);
  }
}

/* =========================================================
   AI RESPONSE
========================================================= */

async function askGemini(
  user,
  memories,
  history,
  userMessage
) {
  if (!gemini) {
    throw new Error("Gemini API is not configured");
  }

  const model = gemini.getGenerativeModel({
    model: "gemini-2.5-flash"
  });

  const systemPrompt = buildSystemPrompt(
    user,
    memories
  );

  const historyText = history
    .map((message) => {
      const role =
        message.role === "assistant"
          ? "Mahi"
          : "User";

      return `${role}: ${message.content}`;
    })
    .join("\n");

  const prompt = `
${systemPrompt}

RECENT CONVERSATION:
${historyText || "No previous conversation."}

USER:
${userMessage}

Now generate the JSON response.
`;

  const result = await model.generateContent(prompt);

  const response = result.response;
  const text = response.text();

  return parseAIResponse(text);
}

/* =========================================================
   SAFE JSON PARSER
========================================================= */

function parseAIResponse(text) {
  let cleaned = text.trim();

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
            }
    };
  } catch {
    return {
      reply: cleaned || "Mujhe samajhne mein dikkat hui.",
      emotion: "neutral",
      state: "ready",
      action: {
        type: "NONE",
        requiresConfirmation: false,
        data: {}
      }
    };
  }
}

/* =========================================================
   CHAT API
========================================================= */

app.post("/api/chat", async (req, res) => {
  try {
    const user = await authenticateUser(req);

    const {
      message,
      conversationId
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

    const cleanMessage = message.trim();

    if (cleanMessage.length > 5000) {
      return res.status(400).json({
        ok: false,
        error: "Message is too long"
      });
    }

    const conversation =
      await getOrCreateConversation(
        user.id,
        conversationId
      );

    const history =
      await getRecentMessages(
        user.id,
        conversation.id
      );

    const memories =
      await getMemories(user.id);

    await saveMessage(
      user.id,
      conversation.id,
      "user",
      cleanMessage
    );

    const aiResponse =
      await askGemini(
        user,
        memories,
        history,
        cleanMessage
      );

    await saveMessage(
      user.id,
      conversation.id,
      "assistant",
      aiResponse.reply
    );

    res.json({
      ok: true,
      conversationId: conversation.id,
      reply: aiResponse.reply,
      emotion: aiResponse.emotion,
      state: aiResponse.state,
      action: aiResponse.action
    });

  } catch (error) {
    console.error("CHAT ERROR:", error);

    res.status(500).json({
      ok: false,
      error: error.message || "Mahi backend error"
    });
  }
});

/* =========================================================
   MEMORY API
========================================================= */

app.get("/api/memories", async (req, res) => {
  try {
    const user = await authenticateUser(req);

    const memories =
      await getMemories(user.id);

    res.json({
      ok: true,
      memories
    });

  } catch (error) {
    res.status(401).json({
      ok: false,
      error: error.message
    });
  }
});

/* =========================================================
   SAVE MEMORY
========================================================= */

app.post("/api/memories", async (req, res) => {
  try {
    const user = await authenticateUser(req);

    const {
      memory_text,
      memory_type = "general"
    } = req.body;

    if (
      typeof memory_text !== "string" ||
      !memory_text.trim()
    ) {
      return res.status(400).json({
        ok: false,
        error: "memory_text is required"
      });
    }

    const { data, error } = await supabase
      .from("memories")
      .insert({
        user_id: user.id,
        memory_text: memory_text.trim(),
        memory_type
      })
      .select()
      .single();

    if (error) {
      throw new Error(error.message);
    }

    res.json({
      ok: true,
      memory: data
    });

  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

/* =========================================================
   DELETE MEMORY
========================================================= */

app.delete("/api/memories/:id", async (req, res) => {
  try {
    const user = await authenticateUser(req);

    const { id } = req.params;

    const { error } = await supabase
      .from("memories")
      .delete()
      .eq("id", id)
      .eq("user_id", user.id);

    if (error) {
      throw new Error(error.message);
    }

    res.json({
      ok: true
    });

  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

/* =========================================================
   START SERVER
========================================================= */

app.listen(PORT, "0.0.0.0", () => {
  console.log(
    `🤖 Mahi AI Backend running on port ${PORT}`
  );

  console.log(
    `Gemini configured: ${!!GEMINI_API_KEY}`
  );

  console.log(
    `Supabase configured: ${
      !!(SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY)
    }`
  );
});
