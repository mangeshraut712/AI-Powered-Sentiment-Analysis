export type FeatureCount = [string, number];

export type BinaryStats = {
  positive_docs: number;
  negative_docs: number;
  total_words: number;
  top_features: {
    positive: FeatureCount[];
    negative: FeatureCount[];
  };
};

export type DatasetStats = {
  binary: BinaryStats;
  emotions: Record<string, number>;
};

/** Dataset snapshots used when the Flask API is not available (GitHub Pages). */
export const DEMO_STATS: DatasetStats = {
  binary: {
    positive_docs: 1000,
    negative_docs: 1000,
    total_words: 18420,
    top_features: {
      positive: [
        ["great", 890],
        ["love", 720],
        ["excellent", 610],
        ["best", 540],
        ["wonderful", 410],
        ["amazing", 380],
        ["good", 350],
        ["perfect", 290],
        ["enjoy", 250],
        ["fun", 210],
      ],
      negative: [
        ["bad", 810],
        ["worst", 640],
        ["terrible", 520],
        ["awful", 470],
        ["poor", 390],
        ["boring", 360],
        ["hate", 330],
        ["waste", 280],
        ["dull", 240],
        ["stupid", 190],
      ],
    },
  },
  emotions: {
    Neutral: 8638,
    Worry: 8459,
    Happiness: 5209,
    Sadness: 5165,
    Love: 3842,
    Surprise: 2187,
    Anger: 1433,
  },
};
