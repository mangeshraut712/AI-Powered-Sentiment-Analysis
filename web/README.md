# Web Interface

Modern web interface for the DSCI-521 Sentiment Analysis project.

## 🎨 Design Philosophy

This interface follows:

- **Minimalism** - Simplicity and elimination of clutter
- **Balance** - Harmony in layout and spacing
- **Apple Design** - Clean, minimal, premium feel
- **shadcn/ui** - Modern component patterns

## 🎯 Features

### Visual Design
- Pure black and white color scheme
- Smooth motion animations (Framer Motion)
- Minimalist typography
- Glass morphism effects
- Modern minimalist aesthetics

### Data Visualizations
- Emotion Distribution Bar Chart
- Proportion Pie Chart
- Emotion Radar Chart
- Model Performance Comparison

### Interactive Demo
- Real-time text analysis
- Animated results display
- Sample text suggestions
- Confidence scores

### Technical Features
- Next.js 16 with App Router
- TypeScript for type safety
- Tailwind CSS for styling
- Recharts for data visualization
- Framer Motion for animations

## 🚀 Getting Started

### Install Dependencies
```bash
cd web
npm install
```

### Run Development Server
```bash
npm run dev
```

### Open in Browser
Navigate to [http://localhost:3000](http://localhost:3000)

## 📁 Project Structure

```
web/
├── src/
│   └── app/
│       ├── globals.css    # Global styles & design tokens
│       ├── layout.tsx     # Root layout
│       └── page.tsx       # Main page component
├── package.json
├── tailwind.config.ts
└── tsconfig.json
```

## 🎨 Color System

### Light Mode
- Background: Pure White (#FFFFFF)
- Foreground: Pure Black (#000000)

### Dark Mode (Default)
- Background: Pure Black (#000000)
- Foreground: Pure White (#FFFFFF)

### Emotion Colors
| Emotion | Color |
|---------|-------|
| Happiness | `#22c55e` (Green) |
| Sadness | `#3b82f6` (Blue) |
| Anger | `#ef4444` (Red) |
| Love | `#ec4899` (Pink) |
| Surprise | `#eab308` (Yellow) |
| Worry | `#8b5cf6` (Purple) |
| Neutral | `#71717a` (Gray) |

## 🔧 Development

### Build for Production
```bash
npm run build
```

### Start Production Server
```bash
npm start
```

### Lint Code
```bash
npm run lint
```

## 🌐 Pages & Sections

### Hero Section
- Animated emotion cycling
- Animated title and description
- Clickable emotion icons
- Smooth entrance animations

### Stats Section
- Key project metrics
- Animated counters
- Clean iconography

### Visualizations Section
- 4 interactive charts
- Responsive layouts
- Tooltip information

### Demo Section
- Text input area
- Sample text buttons
- Analyze button with loading state
- Animated results

### Features Section
- 6 feature cards
- Hover animations
- Icon highlights

### Footer
- Project credits
- Social links
- Copyright

## 📱 Responsive Design

- Mobile-first approach
- Breakpoints: sm, md, lg, xl
- Touch-friendly interactions
- Collapsible elements

## ⚡ Performance

- Next.js Turbopack
- Optimized images
- Code splitting
- Lazy loading

## 🎬 Animations

All animations use Framer Motion:

- `fadeUp` - Elements fade in while moving up
- `scaleIn` - Elements scale in with opacity
- `stagger` - Children animate sequentially
- Spring physics for natural motion

## 🔌 Future Improvements

- [ ] Connect to Python backend API
- [ ] Real-time model predictions
- [ ] Upload CSV for batch analysis
- [ ] Export results functionality
- [ ] User authentication
- [ ] History of analyses
- [ ] Compare multiple texts

## 📝 License

MIT License - Part of DSCI-521 Project
