"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from "recharts";
import {
  Heart, Frown, Angry, Sparkles, AlertCircle, Meh, Smile,
  Brain, TrendingUp, MessageCircle, Zap, ArrowRight, Activity,
  BarChart3, PieChartIcon, Send, Moon, Sun, Github, ChevronDown
} from "lucide-react";

// Emotion data from the actual analysis
const emotionData = [
  { name: "Neutral", value: 8638, color: "#71717a", icon: Meh },
  { name: "Worry", value: 8459, color: "#8b5cf6", icon: AlertCircle },
  { name: "Happiness", value: 5209, color: "#22c55e", icon: Smile },
  { name: "Sadness", value: 5165, color: "#3b82f6", icon: Frown },
  { name: "Love", value: 3842, color: "#ec4899", icon: Heart },
  { name: "Surprise", value: 2187, color: "#eab308", icon: Sparkles },
  { name: "Anger", value: 1433, color: "#ef4444", icon: Angry },
];

const modelPerformance = [
  { name: "Logistic Regression", accuracy: 39.29, precision: 39.85, recall: 39.29, f1: 37.33 },
  { name: "Naive Bayes", accuracy: 37.21, precision: 38.62, recall: 37.21, f1: 33.87 },
];



const radarData = emotionData.map(e => ({
  emotion: e.name,
  fullMark: 10000,
  value: e.value
}));

// Animation variants
const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: "easeOut" as const } }
};

const stagger = {
  visible: { transition: { staggerChildren: 0.1 } }
};

const scaleIn = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.6 } }
};

// Components
function HeroSection() {
  const [currentEmotion, setCurrentEmotion] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentEmotion((prev) => (prev + 1) % emotionData.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const emotion = emotionData[currentEmotion];

  return (
    <section className="min-h-screen flex items-center justify-center relative overflow-hidden">
      {/* Background Grid */}
      <div className="absolute inset-0 grid-pattern opacity-50" />

      {/* Animated Background Orbs */}
      <motion.div
        className="absolute w-[600px] h-[600px] rounded-full opacity-5"
        style={{ background: `radial-gradient(circle, ${emotion.color} 0%, transparent 70%)` }}
        animate={{ scale: [1, 1.2, 1], opacity: [0.05, 0.1, 0.05] }}
        transition={{ duration: 4, repeat: Infinity }}
      />

      <div className="relative z-10 text-center px-6 max-w-5xl mx-auto">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={stagger}
        >
          <motion.p
            variants={fadeUp}
            className="text-sm uppercase tracking-[0.3em] text-muted-foreground mb-4"
          >
            AI-Powered Sentiment Analysis
          </motion.p>

          <motion.h1
            variants={fadeUp}
            className="mb-6"
          >
            Understanding{" "}
            <span className="relative">
              <AnimatePresence mode="wait">
                <motion.span
                  key={emotion.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  style={{ color: emotion.color }}
                  className="inline-block"
                >
                  {emotion.name}
                </motion.span>
              </AnimatePresence>
            </span>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto mb-12 font-light"
          >
            AI-powered emotion detection from 40,000 tweets.
            <br />
            A minimalist approach to understanding human feelings.
          </motion.p>

          {/* Emotion Icons */}
          <motion.div
            variants={fadeUp}
            className="flex justify-center gap-4 mb-12"
          >
            {emotionData.map((e, i) => {
              const EIcon = e.icon;
              return (
                <motion.div
                  key={e.name}
                  whileHover={{ scale: 1.2 }}
                  whileTap={{ scale: 0.9 }}
                  className={`p-3 rounded-full transition-all duration-300 cursor-pointer
                    ${i === currentEmotion ? "ring-2 ring-offset-2 ring-offset-background" : "opacity-40 hover:opacity-100"}`}
                  style={{
                    background: i === currentEmotion ? e.color : "transparent",
                    borderColor: e.color
                  }}
                  onClick={() => setCurrentEmotion(i)}
                >
                  <EIcon
                    size={24}
                    style={{ color: i === currentEmotion ? "white" : e.color }}
                  />
                </motion.div>
              );
            })}
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            variants={fadeUp}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <motion.a
              href="#demo"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="px-8 py-4 bg-foreground text-background rounded-full font-medium flex items-center justify-center gap-2 group"
            >
              Try Demo
              <ArrowRight className="group-hover:translate-x-1 transition-transform" size={18} />
            </motion.a>
            <motion.a
              href="#visualizations"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="px-8 py-4 border border-foreground/20 rounded-full font-medium flex items-center justify-center gap-2"
            >
              View Analytics
              <BarChart3 size={18} />
            </motion.a>
          </motion.div>
        </motion.div>

        {/* Scroll Indicator */}
        <motion.div
          className="absolute bottom-12 left-1/2 -translate-x-1/2"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <ChevronDown className="text-muted-foreground" />
        </motion.div>
      </div>
    </section>
  );
}

function StatsSection() {
  const stats = [
    { value: "40,000", label: "Tweets Analyzed", icon: MessageCircle },
    { value: "7", label: "Emotion Categories", icon: Heart },
    { value: "39.29%", label: "Model Accuracy", icon: TrendingUp },
    { value: "500", label: "TF-IDF Features", icon: Zap },
  ];

  return (
    <section className="py-32 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={stagger}
          className="grid grid-cols-2 md:grid-cols-4 gap-8"
        >
          {stats.map((stat) => (
            <motion.div
              key={stat.label}
              variants={fadeUp}
              className="text-center"
            >
              <stat.icon className="mx-auto mb-4 text-muted-foreground" size={32} />
              <div className="text-4xl md:text-5xl font-extralight mb-2">{stat.value}</div>
              <div className="text-sm text-muted-foreground uppercase tracking-wider">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

function VisualizationsSection() {
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    fetch('http://localhost:5001/api/stats')
      .then(res => res.json())
      .then(data => setStats(data))
      .catch(err => console.error("Stats fetch error:", err));
  }, []);

  const binaryData = stats?.binary ? [
    { name: 'Positive', value: stats.binary.positive_docs, color: '#22c55e' },
    { name: 'Negative', value: stats.binary.negative_docs, color: '#ef4444' }
  ] : [];

  const topPos = stats?.binary?.top_features?.positive?.slice(0, 5).map(([word, count]: any) => ({ name: word, value: count })) || [];
  const topNeg = stats?.binary?.top_features?.negative?.slice(0, 5).map(([word, count]: any) => ({ name: word, value: count })) || [];
  return (
    <section id="visualizations" className="py-32 px-6 bg-secondary/30">
      <div className="max-w-7xl mx-auto">

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={stagger}
          className="text-center mb-20"
        >
          <motion.p variants={fadeUp} className="text-sm uppercase tracking-[0.3em] text-muted-foreground mb-4">
            Data Visualization
          </motion.p>
          <motion.h2 variants={fadeUp}>
            Dataset Insights (Real-time)
          </motion.h2>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Bar Chart */}
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={scaleIn}
            className="bg-card rounded-3xl p-8 border border-border"
          >
            <h3 className="text-xl font-light mb-6 flex items-center gap-2">
              <BarChart3 size={20} />
              Tweet Count by Emotion
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={emotionData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis type="number" stroke="hsl(var(--muted-foreground))" />
                <YAxis dataKey="name" type="category" stroke="hsl(var(--muted-foreground))" width={80} />
                <Tooltip
                  contentStyle={{
                    background: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "12px"
                  }}
                />
                <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                  {emotionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Pie Chart */}
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={scaleIn}
            className="bg-card rounded-3xl p-8 border border-border"
          >
            <h3 className="text-xl font-light mb-6 flex items-center gap-2">
              <PieChartIcon size={20} />
              Emotion Proportion
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={emotionData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {emotionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "12px"
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap justify-center gap-4 mt-4">
              {emotionData.map((e) => (
                <div key={e.name} className="flex items-center gap-2 text-sm">
                  <div className="w-3 h-3 rounded-full" style={{ background: e.color }} />
                  {e.name}
                </div>
              ))}
            </div>
          </motion.div>

          {/* Radar Chart */}
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={scaleIn}
            className="bg-card rounded-3xl p-8 border border-border"
          >
            <h3 className="text-xl font-light mb-6 flex items-center gap-2">
              <Activity size={20} />
              Emotion Radar
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                <PolarGrid stroke="hsl(var(--border))" />
                <PolarAngleAxis dataKey="emotion" stroke="hsl(var(--muted-foreground))" />
                <PolarRadiusAxis stroke="hsl(var(--muted-foreground))" />
                <Radar
                  name="Tweets"
                  dataKey="value"
                  stroke="hsl(var(--foreground))"
                  fill="hsl(var(--foreground))"
                  fillOpacity={0.3}
                />
                <Tooltip
                  contentStyle={{
                    background: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "12px"
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* New: Binary Sentiment Distribution */}
          {stats?.binary && (
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={scaleIn}
              className="bg-card rounded-3xl p-8 border border-border"
            >
              <h3 className="text-xl font-light mb-6 flex items-center gap-2">
                <PieChartIcon size={20} />
                Sentiment Balance (Pos vs Neg)
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={binaryData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {binaryData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "12px"
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex justify-center gap-4 mt-4">
                <div className="flex items-center gap-2 text-sm"><div className="w-3 h-3 rounded-full bg-green-500" /> Positive</div>
                <div className="flex items-center gap-2 text-sm"><div className="w-3 h-3 rounded-full bg-red-500" /> Negative</div>
              </div>
            </motion.div>
          )}

          {/* New: Top Words */}
          {stats?.binary && (
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={scaleIn}
              className="md:col-span-2 bg-card rounded-3xl p-8 border border-border"
            >
              <h3 className="text-xl font-light mb-6 flex items-center gap-2">
                <BarChart3 size={20} />
                Most Frequent Words (Top 5)
              </h3>
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h4 className="text-sm font-medium mb-4 text-green-500">Positive Words</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={topPos} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis type="number" stroke="hsl(var(--muted-foreground))" hide />
                      <YAxis dataKey="name" type="category" stroke="hsl(var(--muted-foreground))" width={80} />
                      <Tooltip contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "12px" }} />
                      <Bar dataKey="value" fill="#22c55e" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div>
                  <h4 className="text-sm font-medium mb-4 text-red-500">Negative Words</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={topNeg} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis type="number" stroke="hsl(var(--muted-foreground))" hide />
                      <YAxis dataKey="name" type="category" stroke="hsl(var(--muted-foreground))" width={80} />
                      <Tooltip contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "12px" }} />
                      <Bar dataKey="value" fill="#ef4444" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </motion.div>
          )}

          {/* Model Performance */}
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={scaleIn}
            className="bg-card rounded-3xl p-8 border border-border"
          >
            <h3 className="text-xl font-light mb-6 flex items-center gap-2">
              <Brain size={20} />
              Model Performance
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelPerformance}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" />
                <YAxis stroke="hsl(var(--muted-foreground))" domain={[0, 50]} />
                <Tooltip
                  contentStyle={{
                    background: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "12px"
                  }}
                />
                <Bar dataKey="accuracy" fill="#22c55e" name="Accuracy %" radius={[8, 8, 0, 0]} />
                <Bar dataKey="f1" fill="#3b82f6" name="F1 Score %" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

function DemoSection() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<null | { emotion: string; confidence: number; sentiment: string }>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const analyzeText = async () => {
    if (!text.trim()) return;

    setIsAnalyzing(true);

    try {
      // Call local Python Backend
      const response = await fetch('http://localhost:5001/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) throw new Error('Network response was not ok');

      const data = await response.json();

      setResult({
        emotion: data.emotion,
        confidence: data.emotion_confidence,
        sentiment: data.sentiment
      });

    } catch (error) {
      console.error("Analysis failed:", error);
      // Fallback for demo if backend not running
      setResult({ emotion: "Error", confidence: 0, sentiment: "Backend Offline" });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getEmotionData = (name: string) => emotionData.find(e => e.name === name) || emotionData[0];

  return (
    <section id="demo" className="py-32 px-6">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={stagger}
          className="text-center mb-16"
        >
          <motion.p variants={fadeUp} className="text-sm uppercase tracking-[0.3em] text-muted-foreground mb-4">
            Live Demo
          </motion.p>
          <motion.h2 variants={fadeUp}>
            Try It Yourself
          </motion.h2>
        </motion.div>

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={scaleIn}
          className="bg-card rounded-3xl p-8 border border-border"
        >
          <div className="relative">
            <textarea
              id="emotion-text-input"
              name="emotion-text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter any text to analyze its emotion..."
              className="w-full h-40 bg-transparent resize-none border-none focus:ring-0 focus:outline-none text-lg placeholder:text-muted-foreground"
              autoComplete="off"
            />
            <div className="zen-line my-4" />
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">
                {text.length} characters
              </span>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={analyzeText}
                disabled={isAnalyzing || !text.trim()}
                className="px-6 py-3 bg-foreground text-background rounded-full font-medium flex items-center gap-2 disabled:opacity-50"
              >
                {isAnalyzing ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    >
                      <Brain size={18} />
                    </motion.div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    Analyze
                    <Send size={18} />
                  </>
                )}
              </motion.button>
            </div>
          </div>

          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-8"
              >
                <div className="zen-line mb-8" />
                <div className="flex items-center justify-center gap-6">
                  {(() => {
                    const data = getEmotionData(result.emotion);
                    const Icon = data.icon;
                    return (
                      <>
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ type: "spring", bounce: 0.5 }}
                          className="w-20 h-20 rounded-full flex items-center justify-center"
                          style={{ background: data.color }}
                        >
                          <Icon size={40} className="text-white" />
                        </motion.div>
                        <div>
                          <div className="text-3xl font-light">{result.emotion}</div>
                          <div className="text-muted-foreground">
                            Confidence: {(result.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className={`text-lg font-medium mt-1 ${result.sentiment === "Positive" ? "text-green-500" : result.sentiment === "Negative" ? "text-red-500" : "text-gray-500"}`}>
                          {result.sentiment.toUpperCase()} SENTIMENT
                        </div>

                      </>
                    );
                  })()}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* Sample Texts */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          className="mt-8 text-center"
        >
          <p className="text-sm text-muted-foreground mb-4">Try these examples:</p>
          <div className="flex flex-wrap justify-center gap-2">
            {[
              "I love this amazing day!",
              "I'm so sad and depressed",
              "This is the worst thing ever!",
              "Wow! I can't believe this!",
              "I'm worried about tomorrow"
            ].map((sample) => (
              <button
                key={sample}
                onClick={() => setText(sample)}
                className="px-4 py-2 rounded-full border border-border hover:bg-secondary transition-colors text-sm"
              >
                {sample}
              </button>
            ))}
          </div>
        </motion.div>
      </div >
    </section >
  );
}

function FeaturesSection() {
  const features = [
    {
      title: "40,000 Tweets",
      description: "Trained on a diverse dataset of real Twitter data to understand human emotions in social media.",
      icon: MessageCircle
    },
    {
      title: "7 Emotions",
      description: "Detects happiness, sadness, anger, love, surprise, worry, and neutral sentiments.",
      icon: Heart
    },
    {
      title: "TF-IDF Features",
      description: "Advanced text vectorization using Term Frequency-Inverse Document Frequency.",
      icon: Brain
    },
    {
      title: "ML Models",
      description: "Logistic Regression and Naive Bayes classifiers for accurate predictions.",
      icon: TrendingUp
    },
    {
      title: "Real-time Analysis",
      description: "Instant emotion detection with confidence scores for any text input.",
      icon: Zap
    },
    {
      title: "Clean Architecture",
      description: "Modular Python package with reusable components and comprehensive documentation.",
      icon: Activity
    }
  ];

  return (
    <section className="py-32 px-6 bg-secondary/30">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={stagger}
          className="text-center mb-20"
        >
          <motion.p variants={fadeUp} className="text-sm uppercase tracking-[0.3em] text-muted-foreground mb-4">
            Features
          </motion.p>
          <motion.h2 variants={fadeUp}>
            Technical Excellence
          </motion.h2>
        </motion.div>

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={stagger}
          className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {features.map((feature) => (
            <motion.div
              key={feature.title}
              variants={fadeUp}
              whileHover={{ y: -5 }}
              className="bg-card rounded-2xl p-8 border border-border group hover:border-foreground/20 transition-colors"
            >
              <feature.icon className="mb-4 text-muted-foreground group-hover:text-foreground transition-colors" size={32} />
              <h3 className="text-xl font-medium mb-2">{feature.title}</h3>
              <p className="text-muted-foreground">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="py-16 px-6 border-t border-border">
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="text-center md:text-left">
            <h3 className="text-xl font-light mb-2">Sentiment Analysis</h3>
            <p className="text-muted-foreground text-sm">
              DSCI-521 Sentiment Analysis Project
            </p>
          </div>

          <div className="flex gap-4">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="p-3 rounded-full border border-border hover:bg-secondary transition-colors"
            >
              <Github size={20} />
            </a>
          </div>
        </div>

        <div className="zen-line my-8" />

        <div className="text-center text-sm text-muted-foreground">
          <p>Built with Next.js, Framer Motion & Recharts</p>
          <p className="mt-2">© 2024 DSCI-521 Group Project • Drexel University</p>
        </div>
      </div>
    </footer>
  );
}

// Main Page
export default function Home() {
  const [isDark, setIsDark] = useState(true);

  useEffect(() => {
    document.documentElement.className = isDark ? "dark" : "";
  }, [isDark]);

  return (
    <main className="min-h-screen">
      {/* Theme Toggle */}
      <motion.button
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        onClick={() => setIsDark(!isDark)}
        className="fixed top-6 right-6 z-50 p-3 rounded-full bg-secondary border border-border hover:bg-accent transition-colors"
      >
        {isDark ? <Sun size={20} /> : <Moon size={20} />}
      </motion.button>

      <HeroSection />
      <StatsSection />
      <VisualizationsSection />
      <DemoSection />
      <FeaturesSection />
      <Footer />
    </main>
  );
}
