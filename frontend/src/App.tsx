import { useState, useRef } from 'react';
import { Upload, FileText, AlertTriangle, ShieldCheck } from 'lucide-react';

const CONTRACT_TYPES = [
  { id: 'employment', label: 'Employment Bond' },
  { id: 'rental', label: 'Rental Lease' },
  { id: 'freelance', label: 'Freelance NDA' },
  { id: 'vc', label: 'VC Term Sheet' },
  { id: 'tos', label: 'Terms of Service' },
  { id: 'general', label: 'General Contract' },
];

type Severity = 'high' | 'medium' | 'low';
type Verdict = 'ACCEPT' | 'NEGOTIATE' | 'REJECT';

type FlaggedClause = {
  clause_title: string;
  plain_english_explanation: string;
  severity: Severity;
};

type SafeClause = {
  clause_title: string;
  plain_english_explanation: string;
};

type AnalyzeResult = {
  risk_score: number;
  compatibility_score: number;
  verdict: Verdict;
  verdict_reason: string;
  red_flags: FlaggedClause[];
  safe_clauses: SafeClause[];
};

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const SCORE_TONE_STYLES = {
  green: {
    badge: 'bg-emerald-100 text-emerald-700',
    text: 'text-emerald-600',
  },
  amber: {
    badge: 'bg-amber-100 text-amber-700',
    text: 'text-amber-600',
  },
  red: {
    badge: 'bg-red-100 text-red-700',
    text: 'text-red-600',
  },
} as const;

const getRiskMeta = (score: number) => {
  if (score <= 30) {
    return { label: 'Safe', description: 'Low objective risk.', tone: 'green' as const };
  }
  if (score <= 60) {
    return { label: 'Caution', description: 'Some risk worth reviewing.', tone: 'amber' as const };
  }
  return { label: 'Dangerous', description: 'High risk and major red flags.', tone: 'red' as const };
};

const getCompatibilityMeta = (score: number) => {
  if (score <= 30) {
    return { label: 'Low Match', description: 'Poor fit with your goals.', tone: 'red' as const };
  }
  if (score <= 60) {
    return { label: 'Mixed Fit', description: 'Some alignment, some conflicts.', tone: 'amber' as const };
  }
  return { label: 'Strong Match', description: 'Aligned with your goals.', tone: 'green' as const };
};

const VERDICT_META = {
  ACCEPT: {
    label: 'ACCEPT',
    description: 'Low risk and strong compatibility.',
    tone: 'green' as const,
  },
  NEGOTIATE: {
    label: 'NEGOTIATE',
    description: 'Some red flags or goal conflicts.',
    tone: 'amber' as const,
  },
  REJECT: {
    label: 'REJECT',
    description: 'High risk or poor compatibility.',
    tone: 'red' as const,
  },
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [inputMode, setInputMode] = useState<'file' | 'text'>('file');
  const [contractText, setContractText] = useState('');
  const [contractType, setContractType] = useState(CONTRACT_TYPES[0].id);
  const [requirements, setRequirements] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    const trimmedText = contractText.trim();
    const trimmedRequirements = requirements.trim();

    if (inputMode === 'file' && !file) {
      setError('Please upload a file.');
      return;
    }

    if (inputMode === 'text' && !trimmedText) {
      setError('Please paste the contract text.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('contract_type', contractType);
    if (inputMode === 'file' && file) {
      formData.append('file', file);
    }
    if (inputMode === 'text') {
      formData.append('text', trimmedText);
    }
    if (trimmedRequirements) {
      formData.append('requirements', trimmedRequirements);
    }

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        let message = `Request failed (${response.status})`;

        if (errorText) {
          try {
            const payload = JSON.parse(errorText) as { detail?: string };
            if (payload.detail) {
              message = `${message}: ${payload.detail}`;
            } else {
              message = `${message}: ${errorText}`;
            }
          } catch {
            message = `${message}: ${errorText}`;
          }
        }

        throw new Error(message);
      }

      const data = (await response.json()) as AnalyzeResult;
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const riskMeta = getRiskMeta(result?.risk_score ?? 0);
  const compatibilityMeta = getCompatibilityMeta(result?.compatibility_score ?? 0);
  const verdictMeta = result ? VERDICT_META[result.verdict] ?? VERDICT_META.NEGOTIATE : VERDICT_META.NEGOTIATE;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText className="text-blue-600 w-6 h-6" />
            <h1 className="text-xl font-bold text-slate-900 tracking-tight">FinePrint</h1>
          </div>
          <div className="text-sm font-medium text-slate-500">Harvey AI for the rest of us</div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8 space-y-8">
        {!result ? (
          <div className="space-y-6 max-w-2xl mx-auto">
            <div className="text-center space-y-2 mb-8">
              <h2 className="text-3xl font-extrabold text-slate-900">Analyze your contract in seconds</h2>
              <p className="text-slate-500">Upload a photo or PDF, or paste text, to instantly identify risks and understand exactly what you are signing.</p>
            </div>

            <div className="flex items-center justify-center">
              <div className="inline-flex rounded-lg border border-slate-200 bg-white p-1 shadow-sm">
                <button
                  type="button"
                  onClick={() => {
                    setInputMode('file');
                    setContractText('');
                    setError(null);
                    setResult(null);
                  }}
                  className={`px-4 py-2 text-sm font-semibold rounded-md transition ${
                    inputMode === 'file'
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Upload File
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setInputMode('text');
                    setFile(null);
                    setError(null);
                    setResult(null);
                  }}
                  className={`px-4 py-2 text-sm font-semibold rounded-md transition ${
                    inputMode === 'text'
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Paste Text
                </button>
              </div>
            </div>

            {inputMode === 'file' ? (
              <div 
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                className="border-2 border-dashed border-slate-300 rounded-2xl bg-white p-10 flex flex-col items-center justify-center gap-4 hover:bg-slate-50 transition-colors cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  className="hidden" 
                  accept="image/*,application/pdf"
                  onChange={handleFileChange}
                />
                <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center text-blue-600 mb-2">
                  <Upload className="w-8 h-8" />
                </div>
                <div className="text-center">
                  <p className="text-lg font-medium text-slate-900">
                    {file ? file.name : "Click or drag to upload"}
                  </p>
                  <p className="text-sm text-slate-500 mt-1">
                    Supports Images (PNG, JPG) and PDFs
                  </p>
                </div>
              </div>
            ) : (
              <div className="border border-slate-200 rounded-2xl bg-white p-6 shadow-sm">
                <label className="block text-sm font-semibold text-slate-700 mb-2">Paste Contract Text</label>
                <textarea
                  value={contractText}
                  onChange={(e) => setContractText(e.target.value)}
                  rows={8}
                  className="w-full resize-y rounded-lg border border-slate-300 px-4 py-3 text-sm text-slate-900 outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Paste the contract text here..."
                />
                <p className="text-xs text-slate-500 mt-2">Tip: You can paste job descriptions or offer letters to review quickly.</p>
              </div>
            )}

            <div className="space-y-3">
              <label className="block text-sm font-semibold text-slate-700">Contract Type</label>
              <select 
                value={contractType}
                onChange={(e) => setContractType(e.target.value)}
                className="w-full bg-white border border-slate-300 rounded-lg px-4 py-3 text-slate-900 outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              >
                {CONTRACT_TYPES.map(type => (
                  <option key={type.id} value={type.id}>{type.label}</option>
                ))}
              </select>
            </div>

            <div className="border border-slate-200 rounded-2xl bg-white p-6 shadow-sm">
              <label className="block text-sm font-semibold text-slate-700 mb-2">Your Requirements</label>
              <textarea
                value={requirements}
                onChange={(e) => setRequirements(e.target.value)}
                rows={4}
                className="w-full resize-y rounded-lg border border-slate-300 px-4 py-3 text-sm text-slate-900 outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Example: I plan to stay 2 years, want to keep my side project, avoid a non-compete, and prefer remote-friendly roles."
              />
              <p className="text-xs text-slate-500 mt-2">Used to compute compatibility and the final verdict.</p>
            </div>

            <button 
              onClick={handleAnalyze}
              disabled={
                isAnalyzing ||
                (inputMode === 'file' ? !file : contractText.trim().length === 0)
              }
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold rounded-lg px-4 py-3 shadow-sm transition-all"
            >
              {isAnalyzing ? 'Analyzing with Gemma 4...' : 'Analyze Contract'}
            </button>
            {error ? (
              <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            ) : null}
          </div>
        ) : (
          <div className="space-y-8">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold">Analysis Results</h2>
              <button 
                onClick={() => {
                  setResult(null);
                  setFile(null);
                  setError(null);
                  setContractText('');
                }}
                className="text-sm font-medium text-blue-600 hover:text-blue-800"
              >
                Start New Analysis
              </button>
            </div>

            <div className="grid gap-4 md:grid-cols-3">
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">Risk Score</h3>
                  <span className={`text-xs font-semibold px-2 py-1 rounded-full ${SCORE_TONE_STYLES[riskMeta.tone].badge}`}>
                    {riskMeta.label}
                  </span>
                </div>
                <div className="mt-4 flex items-baseline gap-2">
                  <span className={`text-4xl font-extrabold ${SCORE_TONE_STYLES[riskMeta.tone].text}`}>
                    {result.risk_score}
                  </span>
                  <span className="text-sm text-slate-500">/ 100</span>
                </div>
                <p className="mt-2 text-sm text-slate-600">{riskMeta.description}</p>
              </div>

              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">Compatibility</h3>
                  <span className={`text-xs font-semibold px-2 py-1 rounded-full ${SCORE_TONE_STYLES[compatibilityMeta.tone].badge}`}>
                    {compatibilityMeta.label}
                  </span>
                </div>
                <div className="mt-4 flex items-baseline gap-2">
                  <span className={`text-4xl font-extrabold ${SCORE_TONE_STYLES[compatibilityMeta.tone].text}`}>
                    {result.compatibility_score}
                  </span>
                  <span className="text-sm text-slate-500">/ 100</span>
                </div>
                <p className="mt-2 text-sm text-slate-600">{compatibilityMeta.description}</p>
              </div>

              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">Final Verdict</h3>
                  <span className={`text-xs font-semibold px-2 py-1 rounded-full ${SCORE_TONE_STYLES[verdictMeta.tone].badge}`}>
                    {verdictMeta.label}
                  </span>
                </div>
                <p className="mt-4 text-sm text-slate-600">{verdictMeta.description}</p>
                <p className="mt-3 text-sm font-semibold text-slate-900">{result.verdict_reason}</p>
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="font-bold flex items-center gap-2 text-slate-800 border-b pb-2">
                <AlertTriangle className="w-5 h-5 text-red-500" />
                Red Flags ({result.red_flags.length})
              </h3>
              <div className="grid gap-4">
                {result.red_flags.map((flag, i) => (
                  <div key={i} className="bg-red-50 border border-red-100 rounded-xl p-5">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-semibold text-red-900">{flag.clause_title}</h4>
                      <span className="bg-red-100 text-red-700 text-xs font-bold px-2 py-1 rounded capitalize">
                        {flag.severity}
                      </span>
                    </div>
                    <p className="text-red-800/80 text-sm leading-relaxed">{flag.plain_english_explanation}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="font-bold flex items-center gap-2 text-slate-800 border-b pb-2">
                <ShieldCheck className="w-5 h-5 text-green-500" />
                Safe Clauses ({result.safe_clauses.length})
              </h3>
              <div className="grid gap-4">
                {result.safe_clauses.map((clause, i) => (
                  <div key={i} className="bg-green-50 border border-green-100 rounded-xl p-5">
                    <h4 className="font-semibold text-green-900 mb-1">{clause.clause_title}</h4>
                    <p className="text-green-800/80 text-sm leading-relaxed">{clause.plain_english_explanation}</p>
                  </div>
                ))}
              </div>
            </div>
            
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
