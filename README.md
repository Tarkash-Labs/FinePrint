# FinePrint

Open-weights multimodal AI that decodes predatory contracts into plain English risk scores — powered by Gemma 4.

## 🚀 Getting Started

Follow these instructions to set up the project. Once set up, you can run the provided `start.ps1` script to easily launch both servers without manual copy-pasting.

### 1. Backend Setup

Open a terminal in the `backend` folder and run the environment setup:

```bash
cd backend
python -m venv ../.venv
..\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

**Important:** Open `backend/.env` and add your `GEMMA_API_KEY`.

### 2. Frontend Setup

Open a terminal in the `frontend` folder and install the dependencies:

```bash
cd frontend
npm install
```

### 3. Running the App

To run the app automatically without copying and pasting commands every time, run the startup script from the root of the project:

```powershell
.\start.ps1
```

This will automatically open two new terminal windows—one running the FastAPI backend and the other running the Vite frontend.

Alternatively, you can run them manually:
- **Backend:** `cd backend` then `python -m uvicorn main:app --reload`
- **Frontend:** `cd frontend` then `npm run dev`
