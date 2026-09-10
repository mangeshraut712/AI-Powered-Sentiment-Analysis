import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "AI-Powered Sentiment Analysis",
  description:
    "Classify binary sentiment and seven emotions from text. Live 2026 demo of the DSCI-521 NLP project.",
  keywords: ["sentiment analysis", "emotion detection", "NLP", "machine learning", "AI"],
  metadataBase: new URL("https://mangeshraut712.github.io/AI-Powered-Sentiment-Analysis/"),
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased`} suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
