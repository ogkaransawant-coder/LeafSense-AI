"use client";

import {
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";

/* ============================================================
   TYPES
============================================================ */

type AffectedArea = {
  percentage?: number;
};

type Diagnosis = {
  prediction?: string;
  raw_class?: string;
  confidence?: number;
  disease_type?: string;
  severity?: string | null;

  affected_area?: AffectedArea;

  visual_evidence?: unknown;

  recommendation?: unknown;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type ChatAssistantProps = {
  diagnosis: Diagnosis | null;
};

/* ============================================================
   SUPPORTED CLASSES
============================================================ */

const SUPPORTED_CLASSES = new Set([
  "healthy",
  "red_spider_mite",
  "rust_level_1",
  "rust_level_2",
  "rust_level_3",
  "rust_level_4",
]);

/* ============================================================
   SAFE VALUE HELPERS
============================================================ */

function safeString(value: unknown): string | undefined {
  if (typeof value === "string") {
    const trimmed = value.trim();

    return trimmed || undefined;
  }

  if (
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return String(value);
  }

  return undefined;
}

function safeStringArray(
  value: unknown
): string[] | undefined {
  if (value == null) {
    return undefined;
  }

  /*
   * Already an array.
   */
  if (Array.isArray(value)) {
    const result = value
      .map((item) => {
        if (typeof item === "string") {
          return item.trim();
        }

        /*
         * Some unhealthy predictions may contain structured
         * visual evidence. Convert it safely to text instead
         * of sending an object to FastAPI.
         */
        if (
          item !== null &&
          typeof item === "object"
        ) {
          try {
            return JSON.stringify(item);
          } catch {
            return "";
          }
        }

        if (
          typeof item === "number" ||
          typeof item === "boolean"
        ) {
          return String(item);
        }

        return "";
      })
      .filter(Boolean);

    return result.length > 0
      ? result
      : undefined;
  }

  /*
   * A single string can also be accepted.
   */
  if (typeof value === "string") {
    const trimmed = value.trim();

    return trimmed
      ? [trimmed]
      : undefined;
  }

  /*
   * Structured object.
   */
  if (typeof value === "object") {
    try {
      return [JSON.stringify(value)];
    } catch {
      return undefined;
    }
  }

  return undefined;
}

function safeNumber(
  value: unknown
): number | undefined {
  if (typeof value !== "number") {
    return undefined;
  }

  if (!Number.isFinite(value)) {
    return undefined;
  }

  return value;
}

function safeAffectedArea(
  value: unknown
): AffectedArea | undefined {
  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value)
  ) {
    return undefined;
  }

  const objectValue =
    value as Record<string, unknown>;

  const percentage = safeNumber(
    objectValue.percentage
  );

  if (percentage === undefined) {
    return undefined;
  }

  return {
    percentage,
  };
}

/* ============================================================
   NORMALIZE DIAGNOSIS FOR BACKEND
============================================================ */

function buildSafeDiagnosis(
  diagnosis: Diagnosis
) {
  const rawClass =
    safeString(diagnosis.raw_class);

  const prediction =
    safeString(diagnosis.prediction);

  /*
   * The backend requires raw_class.
   */
  if (!rawClass) {
    throw new Error(
      "The current analysis does not contain a valid diagnosis class. Please analyze the leaf again."
    );
  }

  /*
   * Prevent invalid classes from reaching the backend.
   */
  if (
    !SUPPORTED_CLASSES.has(
      rawClass.toLowerCase()
    )
  ) {
    throw new Error(
      `Unsupported diagnosis class: ${rawClass}`
    );
  }

  const confidence =
    safeNumber(diagnosis.confidence);

  const diseaseType =
    safeString(diagnosis.disease_type);

  const severity =
    safeString(diagnosis.severity);

  const affectedArea =
    safeAffectedArea(
      diagnosis.affected_area
    );

  const visualEvidence =
    safeStringArray(
      diagnosis.visual_evidence
    );

  const recommendation =
    safeString(
      diagnosis.recommendation
    );

  return {
    prediction,
    raw_class: rawClass,
    confidence,
    disease_type: diseaseType,
    severity,
    affected_area: affectedArea,
    visual_evidence: visualEvidence,
    recommendation,
  };
}

/* ============================================================
   EXTRACT ANSWER FROM BACKEND RESPONSE
============================================================ */

function extractAnswer(data: any): string {
  if (!data) {
    return "";
  }

  if (
    typeof data.answer === "string" &&
    data.answer.trim()
  ) {
    return data.answer.trim();
  }

  if (
    typeof data.response === "string" &&
    data.response.trim()
  ) {
    return data.response.trim();
  }

  if (
    typeof data.message === "string" &&
    data.message.trim()
  ) {
    return data.message.trim();
  }

  if (
    data.data &&
    typeof data.data.answer === "string" &&
    data.data.answer.trim()
  ) {
    return data.data.answer.trim();
  }

  if (
    data.result &&
    typeof data.result.answer === "string" &&
    data.result.answer.trim()
  ) {
    return data.result.answer.trim();
  }

  if (
    Array.isArray(data.candidates)
  ) {
    for (
      const candidate of data.candidates
    ) {
      const text =
        candidate?.content?.parts
          ?.map((part: any) =>
            typeof part?.text === "string"
              ? part.text
              : ""
          )
          .join("")
          .trim() || "";

      if (text) {
        return text;
      }
    }
  }

  return "";
}

/* ============================================================
   BACKEND ERROR EXTRACTION
============================================================ */

function extractBackendError(
  data: any,
  status: number
): string {
  if (!data) {
    return `AI assistant request failed (${status}).`;
  }

  /*
   * FastAPI validation error.
   *
   * Example:
   *
   * {
   *   "detail": [
   *      {
   *        "loc": ["body", "diagnosis", "visual_evidence"],
   *        "msg": "...",
   *        "type": "..."
   *      }
   *   ]
   * }
   */

  if (Array.isArray(data.detail)) {
    const messages =
      data.detail
        .map((item: any) => {
          if (
            typeof item === "string"
          ) {
            return item;
          }

          if (
            item &&
            typeof item.msg === "string"
          ) {
            const location =
              Array.isArray(item.loc)
                ? item.loc.join(".")
                : "";

            return location
              ? `${location}: ${item.msg}`
              : item.msg;
          }

          return "";
        })
        .filter(Boolean);

    if (messages.length > 0) {
      return messages.join(" ");
    }
  }

  if (
    typeof data.detail === "string" &&
    data.detail.trim()
  ) {
    return data.detail.trim();
  }

  if (
    typeof data.error === "string" &&
    data.error.trim()
  ) {
    return data.error.trim();
  }

  if (
    typeof data.message === "string" &&
    data.message.trim()
  ) {
    return data.message.trim();
  }

  return `AI assistant request failed (${status}).`;
}

/* ============================================================
   INLINE MARKDOWN
============================================================ */

function formatInlineMarkdown(
  text: string
) {
  const parts = text.split(
    /(\*\*[^\*]+\*\*)/g
  );

  return parts.map(
    (part, index) => {
      if (
        part.startsWith("**") &&
        part.endsWith("**")
      ) {
        return (
          <strong
            key={index}
            className="font-semibold"
          >
            {part.slice(2, -2)}
          </strong>
        );
      }

      return (
        <span key={index}>
          {part}
        </span>
      );
    }
  );
}

/* ============================================================
   FORMAT ASSISTANT MESSAGE
============================================================ */

function formatAssistantText(
  text: string
) {
  const normalized = text
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .trim();

  const lines =
    normalized.split("\n");

  return lines.map(
    (line, index) => {
      const trimmed =
        line.trim();

      if (!trimmed) {
        return (
          <div
            key={index}
            className="h-2"
          />
        );
      }

      if (
        trimmed.startsWith("### ")
      ) {
        return (
          <div
            key={index}
            className="
              mt-3
              mb-1.5
              font-semibold
              text-zinc-900
            "
          >
            {trimmed.replace(
              /^###\s+/,
              ""
            )}
          </div>
        );
      }

      if (
        trimmed.startsWith("## ")
      ) {
        return (
          <div
            key={index}
            className="
              mt-3
              mb-1.5
              font-semibold
              text-zinc-900
            "
          >
            {trimmed.replace(
              /^##\s+/,
              ""
            )}
          </div>
        );
      }

      if (
        trimmed.startsWith("# ")
      ) {
        return (
          <div
            key={index}
            className="
              mt-3
              mb-1.5
              font-semibold
              text-zinc-900
            "
          >
            {trimmed.replace(
              /^#\s+/,
              ""
            )}
          </div>
        );
      }

      if (
        trimmed.startsWith("- ") ||
        trimmed.startsWith("* ")
      ) {
        return (
          <div
            key={index}
            className="
              ml-1
              flex
              gap-2
            "
          >
            <span>•</span>

            <span>
              {formatInlineMarkdown(
                trimmed.replace(
                  /^[-*]\s+/,
                  ""
                )
              )}
            </span>
          </div>
        );
      }

      const numbered =
        trimmed.match(
          /^(\d+)\.\s+(.+)$/
        );

      if (numbered) {
        return (
          <div
            key={index}
            className="
              ml-1
              flex
              gap-2
            "
          >
            <span>
              {numbered[1]}.
            </span>

            <span>
              {formatInlineMarkdown(
                numbered[2]
              )}
            </span>
          </div>
        );
      }

      return (
        <div key={index}>
          {formatInlineMarkdown(
            trimmed
          )}
        </div>
      );
    }
  );
}

/* ============================================================
   LOCAL FALLBACK
============================================================ */

function buildFallbackAnswer(
  question: string,
  diagnosis: Diagnosis
): string {
  const prediction =
    diagnosis.prediction?.trim() ||
    "Unknown";

  const confidence =
    typeof diagnosis.confidence ===
    "number"
      ? `${diagnosis.confidence.toFixed(
          1
        )}%`
      : "not available";

  const affected =
    typeof diagnosis
      .affected_area
      ?.percentage === "number"
      ? `${diagnosis.affected_area.percentage.toFixed(
          2
        )}%`
      : "not available";

  const normalizedQuestion =
    question.toLowerCase();

  /*
   * HEALTHY
   */

  if (
    prediction.toLowerCase() ===
    "healthy"
  ) {
    return [
      "### Diagnosis Summary",
      "",
      `The model classified this coffee leaf as **Healthy** with a confidence of **${confidence}**.`,
      "",
      `The estimated affected area is **${affected}**.`,
      "",
      "### What this means",
      "",
      "The model did not detect strong visual evidence of the supported coffee-leaf disease classes.",
      "",
      "### What to monitor",
      "",
      "- Check the leaf regularly for new spots or lesions.",
      "- Watch for reddish-brown or orange areas.",
      "- Monitor yellowing or unusual discoloration.",
      "- Compare changes with healthy leaves from the same plant.",
      "",
      "### Important",
      "",
      "This is an AI-assisted visual assessment. It should be treated as a screening result rather than a definitive agricultural diagnosis.",
    ].join("\n");
  }

  /*
   * DIAGNOSIS QUESTION
   */

  if (
    normalizedQuestion.includes(
      "what does"
    ) ||
    normalizedQuestion.includes(
      "mean"
    ) ||
    normalizedQuestion.includes(
      "diagnosis"
    )
  ) {
    return [
      "### Diagnosis Summary",
      "",
      `The model classified this coffee leaf as **${prediction}** with a confidence of **${confidence}**.`,
      "",
      `The estimated affected area is **${affected}**.`,
      "",
      "### What this means",
      "",
      "The image contains visual patterns that the model associated with this diagnosis.",
      "",
      "### What to monitor",
      "",
      "- Watch whether the affected area increases.",
      "- Check for new spots, discoloration, or lesions.",
      "- Inspect nearby leaves for similar symptoms.",
      "- Monitor the plant regularly for changes.",
      "",
      "### Important",
      "",
      "This is an AI-assisted visual assessment and should not replace an expert agricultural diagnosis.",
    ].join("\n");
  }

  /*
   * GENERAL QUESTION
   */

  return [
    "### LeafSense AI",
    "",
    `The current diagnosis is **${prediction}**.`,
    "",
    `Confidence: **${confidence}**`,
    "",
    `Estimated affected area: **${affected}**`,
    "",
    "Ask me about the diagnosis, symptoms, monitoring, or recommended next steps.",
  ].join("\n");
}

/* ============================================================
   COMPONENT
============================================================ */

export function ChatAssistant({
  diagnosis,
}: ChatAssistantProps) {
  /* ==========================================================
     STATE
  ========================================================== */

  const [
    isOpen,
    setIsOpen,
  ] = useState(false);

  const [
    question,
    setQuestion,
  ] = useState("");

  const [
    messages,
    setMessages,
  ] = useState<ChatMessage[]>([]);

  const [
    loading,
    setLoading,
  ] = useState(false);

  const [
    error,
    setError,
  ] = useState("");

  const messagesEndRef =
    useRef<HTMLDivElement | null>(
      null
    );

  const inputRef =
    useRef<HTMLTextAreaElement | null>(
      null
    );

  /* ==========================================================
     AUTO SCROLL
  ========================================================== */

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView(
      {
        behavior: "smooth",
      }
    );
  }, [
    messages,
    loading,
  ]);

  /* ==========================================================
     RESET CHAT WHEN NEW ANALYSIS IS PERFORMED
  ========================================================== */

  useEffect(() => {
    setMessages([]);
    setQuestion("");
    setError("");
    setIsOpen(false);
  }, [
    diagnosis?.raw_class,
    diagnosis?.prediction,
    diagnosis?.confidence,
  ]);

  /* ==========================================================
     SEND MESSAGE
  ========================================================== */

  const sendMessage = async (
    customQuestion?: string
  ) => {
    const trimmedQuestion = (
      customQuestion ??
      question
    ).trim();

    if (!trimmedQuestion) {
      return;
    }

    if (loading) {
      return;
    }

    if (!diagnosis) {
      setError(
        "Please analyze a coffee leaf first."
      );

      return;
    }

    setError("");

    /*
     * Validate and normalize diagnosis
     * BEFORE sending it to FastAPI.
     */
    let safeDiagnosis;

    try {
      safeDiagnosis =
        buildSafeDiagnosis(
          diagnosis
        );
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Invalid diagnosis data.";

      setError(message);

      return;
    }

    const userMessage: ChatMessage = {
      role: "user",
      content: trimmedQuestion,
    };

    /*
     * Keep the user message in the
     * conversation history sent to backend.
     */
    const updatedMessages = [
      ...messages,
      userMessage,
    ];

    setMessages(
      updatedMessages
    );

    setQuestion("");
    setLoading(true);

    try {
      /* ======================================================
         API REQUEST
      ====================================================== */

      const response =
        await fetch(
          "/api/chat",
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body: JSON.stringify({
              question:
                trimmedQuestion,

              diagnosis:
                safeDiagnosis,

              history:
                messages,
            }),
          }
        );

      /* ======================================================
         CONTENT TYPE
      ====================================================== */

      const contentType =
        response.headers.get(
          "content-type"
        ) || "";

      let data: any = null;

      if (
        contentType.includes(
          "application/json"
        )
      ) {
        data =
          await response.json();
      } else {
        const rawText =
          await response.text();

        throw new Error(
          `Backend returned an unexpected response (${response.status}): ${
            rawText.slice(
              0,
              300
            ) ||
            "empty response"
          }`
        );
      }

      /* ======================================================
         ERROR RESPONSE
      ====================================================== */

      if (!response.ok) {
        throw new Error(
          extractBackendError(
            data,
            response.status
          )
        );
      }

      /* ======================================================
         EXTRACT ANSWER
      ====================================================== */

      let answer =
        extractAnswer(data);

      /* ======================================================
         CHECK INCOMPLETE RESPONSE
      ====================================================== */

      const normalizedAnswer =
        answer
          .toLowerCase()
          .replace(
            /\s+/g,
            " "
          )
          .trim();

      const looksIncomplete =
        !answer ||
        answer.length < 30 ||
        normalizedAnswer ===
          "header/summary:" ||
        normalizedAnswer.endsWith(
          "the ai model classified"
        ) ||
        normalizedAnswer.endsWith(
          "the ai model classified the"
        );

      /* ======================================================
         LOCAL FALLBACK
      ====================================================== */

      if (looksIncomplete) {
        answer =
          buildFallbackAnswer(
            trimmedQuestion,
            diagnosis
          );
      }

      /* ======================================================
         ASSISTANT MESSAGE
      ====================================================== */

      const assistantMessage:
        ChatMessage = {
        role: "assistant",
        content: answer,
      };

      setMessages([
        ...updatedMessages,
        assistantMessage,
      ]);
    } catch (err) {
      console.error(
        "Chat error:",
        err
      );

      const errorMessage =
        err instanceof Error
          ? err.message
          : "Something went wrong while contacting the AI assistant.";

      setError(
        errorMessage
      );

      /*
       * Remove the failed user message
       * so the UI does not show a message
       * that was never successfully sent.
       */
      setMessages(
        messages
      );
    } finally {
      setLoading(false);
    }
  };

  /* ==========================================================
     KEYBOARD HANDLER
  ========================================================== */

  const handleKeyDown = (
    event: KeyboardEvent<HTMLTextAreaElement>
  ) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();

      sendMessage();
    }
  };

  /* ==========================================================
     CLEAR CHAT
  ========================================================== */

  const clearChat = () => {
    setMessages([]);
    setError("");
    setQuestion("");

    setTimeout(() => {
      inputRef.current?.focus();
    }, 50);
  };

  /* ==========================================================
     CLOSE CHAT
  ========================================================== */

  const closeChat = () => {
    setIsOpen(false);
    setQuestion("");
    setError("");
  };

  /* ==========================================================
     NO DIAGNOSIS = NO CHATBOT
  ========================================================== */

  if (!diagnosis) {
    return null;
  }

  /* ==========================================================
     FLOATING CHATBOT
  ========================================================== */

  return (
    <>
      {/* ======================================================
          FLOATING BUTTON
      ====================================================== */}

      {!isOpen && (
        <button
          type="button"
          onClick={() =>
            setIsOpen(true)
          }
          aria-label="Open LeafSense AI"
          title="Ask LeafSense AI"
          className="
            fixed
            right-6
            bottom-6
            z-[9999]

            flex
            h-16
            w-16
            items-center
            justify-center

            rounded-full

            border
            border-white

            bg-zinc-900

            text-white

            shadow-[0_10px_35px_rgba(0,0,0,0.30)]

            transition-all
            duration-200

            hover:scale-105
            hover:bg-zinc-800
            hover:shadow-[0_14px_40px_rgba(0,0,0,0.35)]

            active:scale-95
          "
        >
          {/* Online indicator */}

          <span
            className="
              absolute
              right-1
              top-1

              h-3
              w-3

              rounded-full

              bg-emerald-400

              ring-2
              ring-white
            "
          />

          {/* Robot icon */}

          <span
            className="
              text-[30px]
              leading-none
            "
          >
            🤖
          </span>
        </button>
      )}

      {/* ======================================================
          CHAT WINDOW
      ====================================================== */}

      {isOpen && (
        <div
          className="
            fixed
            right-5
            bottom-5

            z-[9999]

            flex
            h-[min(680px,calc(100vh-40px))]
            w-[min(430px,calc(100vw-24px))]

            flex-col

            overflow-hidden

            rounded-2xl

            border
            border-zinc-200

            bg-white

            shadow-[0_20px_70px_rgba(0,0,0,0.30)]

            animate-in
            fade-in
            slide-in-from-bottom-3
            duration-200
          "
        >
          {/* ==================================================
              HEADER
          ================================================== */}

          <div
            className="
              flex
              shrink-0
              items-center
              justify-between

              bg-zinc-900

              px-4
              py-3.5

              text-white
            "
          >
            <div
              className="
                flex
                items-center
                gap-3
              "
            >
              {/* Bot avatar */}

              <div
                className="
                  relative

                  flex
                  h-10
                  w-10
                  shrink-0

                  items-center
                  justify-center

                  rounded-full

                  bg-white

                  text-xl
                "
              >
                🤖

                <span
                  className="
                    absolute
                    bottom-0
                    right-0

                    h-2.5
                    w-2.5

                    rounded-full

                    bg-emerald-400

                    ring-2
                    ring-white
                  "
                />
              </div>

              {/* Title */}

              <div>
                <div
                  className="
                    text-sm
                    font-semibold
                  "
                >
                  LeafSense AI
                </div>

                <div
                  className="
                    mt-0.5
                    text-[11px]
                    text-zinc-300
                  "
                >
                  AI diagnosis assistant
                </div>
              </div>
            </div>

            {/* Controls */}

            <div
              className="
                flex
                items-center
                gap-1
              "
            >
              {messages.length > 0 && (
                <button
                  type="button"
                  onClick={
                    clearChat
                  }
                  className="
                    rounded-lg
                    px-2.5
                    py-1.5

                    text-xs
                    text-zinc-300

                    transition

                    hover:bg-zinc-800
                    hover:text-white
                  "
                >
                  Clear
                </button>
              )}

              <button
                type="button"
                onClick={
                  closeChat
                }
                aria-label="Close LeafSense AI"
                className="
                  flex
                  h-8
                  w-8

                  items-center
                  justify-center

                  rounded-full

                  text-xl
                  leading-none
                  text-zinc-300

                  transition

                  hover:bg-zinc-800
                  hover:text-white
                "
              >
                ×
              </button>
            </div>
          </div>

          {/* ==================================================
              DIAGNOSIS BAR
          ================================================== */}

          <div
            className="
              shrink-0

              border-b
              border-zinc-200

              bg-zinc-50

              px-4
              py-3
            "
          >
            <div
              className="
                mb-1

                text-[10px]
                font-medium
                uppercase
                tracking-wide

                text-zinc-400
              "
            >
              Current diagnosis
            </div>

            <div
              className="
                flex
                flex-wrap
                items-center
                gap-x-3
                gap-y-1
              "
            >
              <span
                className="
                  text-sm
                  font-semibold
                  text-zinc-900
                "
              >
                {diagnosis.prediction ||
                  "Unknown"}
              </span>

              {typeof diagnosis.confidence ===
                "number" && (
                <span
                  className="
                    text-xs
                    text-zinc-500
                  "
                >
                  {diagnosis.confidence.toFixed(
                    1
                  )}
                  % confidence
                </span>
              )}

              {typeof diagnosis
                .affected_area
                ?.percentage ===
                "number" && (
                <span
                  className="
                    text-xs
                    text-zinc-500
                  "
                >
                  {diagnosis.affected_area.percentage.toFixed(
                    2
                  )}
                  % affected
                </span>
              )}
            </div>
          </div>

          {/* ==================================================
              CHAT BODY
          ================================================== */}

          <div
            className="
              flex-1

              overflow-y-auto

              bg-zinc-50

              p-4

              scrollbar-thin
            "
          >
            {/* =================================================
                EMPTY STATE
            ================================================= */}

            {messages.length ===
              0 && (
              <div
                className="
                  flex
                  min-h-full

                  flex-col
                  items-center
                  justify-center

                  px-4

                  text-center
                "
              >
                <div
                  className="
                    mb-4

                    flex
                    h-16
                    w-16

                    items-center
                    justify-center

                    rounded-full

                    bg-zinc-900

                    text-3xl

                    shadow-lg
                  "
                >
                  🤖
                </div>

                <h3
                  className="
                    text-base
                    font-semibold
                    text-zinc-900
                  "
                >
                  Ask LeafSense AI
                </h3>

                <p
                  className="
                    mt-1

                    max-w-[300px]

                    text-xs
                    leading-5

                    text-zinc-500
                  "
                >
                  Ask questions about
                  your coffee leaf
                  diagnosis, symptoms,
                  monitoring, or next
                  steps.
                </p>

                <div
                  className="
                    mt-5

                    flex
                    flex-wrap

                    justify-center

                    gap-2
                  "
                >
                  {[
                    "What does this diagnosis mean?",
                    "How serious is this?",
                    "What should I monitor?",
                  ].map(
                    (suggestion) => (
                      <button
                        key={
                          suggestion
                        }
                        type="button"
                        disabled={
                          loading
                        }
                        onClick={() =>
                          sendMessage(
                            suggestion
                          )
                        }
                        className="
                          rounded-full

                          border
                          border-zinc-300

                          bg-white

                          px-3
                          py-2

                          text-xs
                          text-zinc-700

                          shadow-sm

                          transition

                          hover:border-zinc-400
                          hover:bg-zinc-100

                          disabled:cursor-not-allowed
                          disabled:opacity-50
                        "
                      >
                        {
                          suggestion
                        }
                      </button>
                    )
                  )}
                </div>
              </div>
            )}

            {/* =================================================
                MESSAGES
            ================================================= */}

            {messages.map(
              (
                message,
                index
              ) => {
                const isUser =
                  message.role ===
                  "user";

                return (
                  <div
                    key={`${message.role}-${index}`}
                    className={`
                      mb-3
                      flex
                      ${
                        isUser
                          ? "justify-end"
                          : "justify-start"
                      }
                    `}
                  >
                    {!isUser && (
                      <div
                        className="
                          mr-2
                          mt-1

                          flex
                          h-7
                          w-7
                          shrink-0

                          items-center
                          justify-center

                          rounded-full

                          bg-zinc-900

                          text-sm
                        "
                      >
                        🤖
                      </div>
                    )}

                    <div
                      className={`
                        max-w-[82%]

                        rounded-2xl

                        px-3.5
                        py-2.5

                        text-sm
                        leading-6

                        break-words

                        shadow-sm

                        ${
                          isUser
                            ? `
                              rounded-br-md
                              bg-zinc-900
                              text-white
                            `
                            : `
                              rounded-bl-md
                              border
                              border-zinc-200
                              bg-white
                              text-zinc-800
                            `
                        }
                      `}
                    >
                      {isUser ? (
                        <div
                          className="
                            whitespace-pre-wrap
                          "
                        >
                          {
                            message.content
                          }
                        </div>
                      ) : (
                        formatAssistantText(
                          message.content
                        )
                      )}
                    </div>
                  </div>
                );
              }
            )}

            {/* =================================================
                LOADING
            ================================================= */}

            {loading && (
              <div
                className="
                  mb-3
                  flex
                  justify-start
                "
              >
                <div
                  className="
                    mr-2
                    mt-1

                    flex
                    h-7
                    w-7
                    shrink-0

                    items-center
                    justify-center

                    rounded-full

                    bg-zinc-900

                    text-sm
                  "
                >
                  🤖
                </div>

                <div
                  className="
                    rounded-2xl
                    rounded-bl-md

                    border
                    border-zinc-200

                    bg-white

                    px-4
                    py-3

                    shadow-sm
                  "
                >
                  <div
                    className="
                      flex
                      items-center
                      gap-2
                    "
                  >
                    <span
                      className="
                        text-xs
                        text-zinc-500
                      "
                    >
                      Thinking
                    </span>

                    <div
                      className="
                        flex
                        gap-1
                      "
                    >
                      <span
                        className="
                          h-1.5
                          w-1.5

                          animate-bounce

                          rounded-full

                          bg-zinc-400
                        "
                      />

                      <span
                        className="
                          h-1.5
                          w-1.5

                          animate-bounce

                          rounded-full

                          bg-zinc-400

                          [animation-delay:150ms]
                        "
                      />

                      <span
                        className="
                          h-1.5
                          w-1.5

                          animate-bounce

                          rounded-full

                          bg-zinc-400

                          [animation-delay:300ms]
                        "
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div
              ref={
                messagesEndRef
              }
            />
          </div>

          {/* ==================================================
              ERROR
          ================================================== */}

          {error && (
            <div
              className="
                shrink-0

                border-t
                border-red-200

                bg-red-50

                px-4
                py-2.5

                text-xs
                leading-5

                text-red-700
              "
            >
              {error}
            </div>
          )}

          {/* ==================================================
              INPUT AREA
          ================================================== */}

          <div
            className="
              shrink-0

              border-t
              border-zinc-200

              bg-white

              p-3
            "
          >
            <div
              className="
                flex
                items-end
                gap-2
              "
            >
              <textarea
                ref={
                  inputRef
                }
                value={
                  question
                }
                onChange={(
                  event
                ) =>
                  setQuestion(
                    event.target.value
                  )
                }
                onKeyDown={
                  handleKeyDown
                }
                disabled={
                  loading
                }
                rows={2}
                placeholder="Ask about this diagnosis..."
                className="
                  min-h-[48px]
                  max-h-[120px]

                  flex-1

                  resize-none

                  rounded-xl

                  border
                  border-zinc-300

                  bg-white

                  px-3
                  py-2.5

                  text-sm
                  leading-5
                  text-zinc-900

                  outline-none

                  transition

                  placeholder:text-zinc-400

                  focus:border-zinc-500

                  focus:ring-2
                  focus:ring-zinc-200

                  disabled:cursor-not-allowed
                  disabled:bg-zinc-100
                "
              />

              <button
                type="button"
                onClick={() =>
                  sendMessage()
                }
                disabled={
                  !question.trim() ||
                  loading
                }
                className="
                  flex
                  h-11
                  shrink-0

                  items-center
                  justify-center

                  rounded-xl

                  bg-zinc-900

                  px-4

                  text-sm
                  font-medium

                  text-white

                  transition

                  hover:bg-zinc-700

                  active:scale-95

                  disabled:cursor-not-allowed
                  disabled:opacity-40
                "
              >
                {loading
                  ? "..."
                  : "Send"}
              </button>
            </div>

            <div
              className="
                mt-1.5

                text-[10px]

                text-zinc-400
              "
            >
              Enter to send · Shift +
              Enter for a new line
            </div>
          </div>
        </div>
      )}
    </>
  );
}