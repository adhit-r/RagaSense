# RagaSense Website

Revolutionary AI platform for Indian classical music classification and generation.

## 🎵 Features

- **YuE Foundation Model**: State-of-the-art 2025 music foundation model
- **1,616+ Raga Dataset**: Comprehensive collection of Carnatic and Hindustani ragas
- **Real-time Classification**: Instant raga identification with 95%+ accuracy
- **Interactive Demo**: Upload audio files for classification
- **Research Documentation**: Technical details and methodology

## 🚀 Deployment

### Vercel (Recommended)

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Deploy:
```bash
vercel --prod
```

### Netlify

1. Install Netlify CLI:
```bash
npm i -g netlify-cli
```

2. Deploy:
```bash
netlify deploy --prod --dir .
```

### GitHub Pages

1. Push to GitHub repository
2. Enable GitHub Pages in repository settings
3. Select source branch (usually `main`)

## 🛠️ Local Development

1. Start local server:
```bash
python3 server.py
```

2. Open http://localhost:8081

## 📁 Project Structure

```
website/
├── index.html          # Main homepage
├── demo.html           # Interactive demo
├── research.html       # Research documentation
├── terminal-style.css  # Professional styling
├── server.py          # Local development server
├── vercel.json        # Vercel configuration
└── package.json       # Project metadata
```

## 🎯 Author

**Adhithya Rajasekaran** ([@adhit-r](https://github.com/adhit-r))

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Links

- **Live Site**: https://ragasense.vercel.app
- **GitHub**: https://github.com/adhit-r/RagaSense
- **Demo**: https://ragasense.vercel.app/demo.html
- **Research**: https://ragasense.vercel.app/research.html