export const EMOTION_NAMES = [
  "Neutral",
  "Worry",
  "Happiness",
  "Sadness",
  "Love",
  "Surprise",
  "Anger",
] as const;

export type EmotionName = (typeof EMOTION_NAMES)[number];
export type SentimentLabel = "Positive" | "Negative" | "Neutral";
export type AnalysisSource = "api" | "demo";

export type AnalysisResult = {
  emotion: EmotionName;
  confidence: number;
  sentiment: SentimentLabel;
  source: AnalysisSource;
};

const EMOTION_LEXICON: Record<Exclude<EmotionName, "Neutral">, string[]> = {
  Happiness: [
    "happy",
    "happiness",
    "glad",
    "joy",
    "joyful",
    "wonderful",
    "great",
    "excellent",
    "awesome",
    "delighted",
    "cheerful",
    "smile",
    "fun",
    "good",
    "amazing day",
  ],
  Love: [
    "love",
    "loved",
    "adore",
    "adorable",
    "heart",
    "romance",
    "romantic",
    "sweetheart",
    "cherish",
    "affection",
    "kiss",
  ],
  Sadness: [
    "sad",
    "sadness",
    "depressed",
    "depression",
    "unhappy",
    "cry",
    "crying",
    "lonely",
    "miserable",
    "heartbroken",
    "grief",
    "tears",
  ],
  Anger: [
    "angry",
    "anger",
    "hate",
    "furious",
    "rage",
    "annoyed",
    "worst",
    "terrible",
    "awful",
    "disgusting",
    "mad",
    "furious",
  ],
  Surprise: [
    "wow",
    "unbelievable",
    "shocked",
    "shocking",
    "can't believe",
    "cannot believe",
    "unexpected",
    "whoa",
    "surprised",
    "surprise",
    "amazing",
  ],
  Worry: [
    "worried",
    "worry",
    "anxious",
    "anxiety",
    "nervous",
    "afraid",
    "scared",
    "concern",
    "stress",
    "stressed",
    "tomorrow",
    "uncertain",
  ],
};

const POSITIVE_WORDS = [
  "love",
  "great",
  "amazing",
  "wonderful",
  "excellent",
  "good",
  "happy",
  "best",
  "awesome",
  "delight",
  "perfect",
  "joy",
];

const NEGATIVE_WORDS = [
  "hate",
  "worst",
  "terrible",
  "awful",
  "bad",
  "sad",
  "angry",
  "fail",
  "failure",
  "depressed",
  "horrible",
  "disgusting",
];

function tokenize(text: string): string {
  return ` ${text.toLowerCase().replace(/[^a-z0-9'\s]/g, " ")} `;
}

function countHits(haystack: string, terms: string[]): number {
  return terms.reduce((score, term) => {
    const needle = ` ${term} `;
    if (haystack.includes(needle) || haystack.includes(term)) {
      return score + (term.includes(" ") ? 2 : 1);
    }
    return score;
  }, 0);
}

function clampConfidence(raw: number): number {
  return Math.min(0.92, Math.max(0.42, raw));
}

export function normalizeEmotion(name: string): EmotionName {
  const key = name.trim().toLowerCase();
  switch (key) {
    case "neutral":
      return "Neutral";
    case "worry":
      return "Worry";
    case "happiness":
    case "happy":
    case "fun":
    case "enthusiasm":
      return "Happiness";
    case "sadness":
    case "sad":
      return "Sadness";
    case "love":
      return "Love";
    case "surprise":
    case "surprised":
      return "Surprise";
    case "anger":
    case "angry":
    case "hate":
      return "Anger";
    default:
      return "Neutral";
  }
}

export function normalizeSentiment(name: string): SentimentLabel {
  const key = name.trim().toLowerCase();
  switch (key) {
    case "positive":
    case "pos":
      return "Positive";
    case "negative":
    case "neg":
      return "Negative";
    case "neutral":
      return "Neutral";
    default:
      return "Neutral";
  }
}

export function analyzeWithLexicon(text: string): AnalysisResult {
  const haystack = tokenize(text);
  const scores: Record<EmotionName, number> = {
    Neutral: 0.15,
    Worry: 0,
    Happiness: 0,
    Sadness: 0,
    Love: 0,
    Surprise: 0,
    Anger: 0,
  };

  (Object.keys(EMOTION_LEXICON) as Array<Exclude<EmotionName, "Neutral">>).forEach(
    (emotion) => {
      scores[emotion] = countHits(haystack, EMOTION_LEXICON[emotion]);
    }
  );

  let emotion: EmotionName = "Neutral";
  let best = scores.Neutral;
  for (const name of EMOTION_NAMES) {
    if (scores[name] > best) {
      best = scores[name];
      emotion = name;
    }
  }

  const pos = countHits(haystack, POSITIVE_WORDS);
  const neg = countHits(haystack, NEGATIVE_WORDS);
  let sentiment: SentimentLabel;
  if (pos > neg) {
    sentiment = "Positive";
  } else if (neg > pos) {
    sentiment = "Negative";
  } else {
    switch (emotion) {
      case "Happiness":
      case "Love":
        sentiment = "Positive";
        break;
      case "Sadness":
      case "Anger":
      case "Worry":
        sentiment = "Negative";
        break;
      case "Neutral":
      case "Surprise":
        sentiment = "Neutral";
        break;
      default: {
        const _exhaustive: never = emotion;
        sentiment = _exhaustive;
      }
    }
  }

  const wordCount = Math.max(1, text.trim().split(/\s+/).length);
  const confidence = clampConfidence(0.45 + best / (wordCount + 2) + Math.abs(pos - neg) * 0.08);

  return {
    emotion,
    confidence,
    sentiment,
    source: "demo",
  };
}

function getApiBase(): string | null {
  const configured = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (configured) {
    return configured.replace(/\/$/, "");
  }
  if (typeof window !== "undefined" && window.location.hostname === "localhost") {
    return "http://localhost:5001";
  }
  return null;
}

export async function analyzeTextWithFallback(text: string): Promise<AnalysisResult> {
  const api = getApiBase();
  if (api) {
    try {
      const response = await fetch(`${api}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (response.ok) {
        const data = (await response.json()) as {
          emotion?: string;
          emotion_confidence?: number;
          sentiment?: string;
        };
        return {
          emotion: normalizeEmotion(data.emotion ?? "Neutral"),
          confidence: Number(data.emotion_confidence ?? 0.5),
          sentiment: normalizeSentiment(data.sentiment ?? "Neutral"),
          source: "api",
        };
      }
    } catch {
      // Fall through to the on-device lexicon so GitHub Pages still works.
    }
  }
  return analyzeWithLexicon(text);
}

export async function fetchStatsWithFallback<T>(fallback: T): Promise<T> {
  const api = getApiBase();
  if (!api) {
    return fallback;
  }
  try {
    const response = await fetch(`${api}/api/stats`);
    if (!response.ok) {
      return fallback;
    }
    return (await response.json()) as T;
  } catch {
    return fallback;
  }
}
