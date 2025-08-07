┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend UI   │────▶│  Flask/FastAPI   │────▶│ Python Backend  │
│  (React/Vue)    │◀────│   Web Server     │◀────│ (Core Logic)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                ┌──────────────────────────┼──────────────────────────┐
                                │                          │                          │
                        ┌───────▼────────┐      ┌─────────▼────────┐      ┌──────────▼────────┐
                        │ LLM APIs       │      │ PDF Processor    │      │ Web Search API    │
                        │ • OpenAI       │      │ • pdfplumber     │      │ • Tavily          │
                        │ • Anthropic    │      │ • PyPDF2         │      └───────────────────┘
                        │ • Google       │      └──────────────────┘
                        │ • OpenRouter   │
                        └────────────────┘

persuasion-optimizer/
├── backend/
│   ├── app.py                 # FastAPI/Flask application
│   ├── core/
│   │   ├── __init__.py
│   │   ├── optimizer.py       # PersuasionOptimizer class
│   │   ├── profiler.py        # UserProfiler class
│   │   ├── tournament.py      # TournamentSelector class
│   │   ├── pdf_cleaner.py     # PDFTextCleaner class
│   │   └── models.py          # Data models
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # API endpoints
│   ├── config/
│   │   └── settings.py        # Configuration
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
├── data/
│   └── pdfs/                  # PDF storage
├── .env                       # Environment variables
└── README.md