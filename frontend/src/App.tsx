import { useState, useRef } from 'react';
import { Upload, FileText, AlertTriangle, ShieldCheck, ChevronDown, ChevronUp, CheckCircle2, XCircle, Copy, Download, Loader2, Building, User } from 'lucide-react';

const CONTRACT_TYPES = [
  { id: 'employment', label: 'Employment Bond' },
  { id: 'internship', label: 'Internship Agreement' },
  { id: 'rental', label: 'Rental Lease' },
  { id: 'freelance', label: 'Freelance NDA' },
  { id: 'vc', label: 'VC Term Sheet' },
  { id: 'tos', label: 'Terms of Service' },
  { id: 'general', label: 'General Contract' },
];

type Severity = 'high' | 'medium' | 'low';
type Verdict = 'ACCEPT' | 'NEGOTIATE' | 'REJECT';

type RequirementMatch = { requirement: string; met: boolean; explanation: string; };
type FlaggedClause = { clause_title: string; clause_text: string; plain_english_explanation: string; negotiation_tip: string; severity: Severity; };
type SafeClause = { clause_title: string; plain_english_explanation: string; };

type AnalyzeResult = {
  risk_score: number | null;
  compatibility_score: number | null;
  verdict: Verdict | null;
  verdict_reason: string | null;
  requirement_breakdown: RequirementMatch[];
  red_flags: FlaggedClause[];
  safe_clauses: SafeClause[];
  negotiation_email: string | null;
};

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const SCORE_TONE_STYLES = {
  green: { badge: 'bg-emerald-100 text-emerald-700', text: 'text-emerald-600' },
  amber: { badge: 'bg-amber-100 text-amber-700', text: 'text-amber-600' },
  red: { badge: 'bg-red-100 text-red-700', text: 'text-red-600' },
  gray: { badge: 'bg-slate-100 text-slate-700', text: 'text-slate-400' },
} as const;

const getRiskMeta = (score: number | null) => {
  if (score === null) return { label: 'Analyzing...', description: 'Waiting for Gemma...', tone: 'gray' as const };
  if (score <= 30) return { label: 'Safe', description: 'Low objective risk.', tone: 'green' as const };
  if (score <= 60) return { label: 'Caution', description: 'Some risk worth reviewing.', tone: 'amber' as const };
  return { label: 'Dangerous', description: 'High risk and major red flags.', tone: 'red' as const };
};

const getCompatibilityMeta = (score: number | null) => {
  if (score === null) return { label: 'Analyzing...', description: 'Waiting for Gemma...', tone: 'gray' as const };
  if (score <= 30) return { label: 'Low Match', description: 'Poor fit with your goals.', tone: 'red' as const };
  if (score <= 60) return { label: 'Mixed Fit', description: 'Some alignment, some conflicts.', tone: 'amber' as const };
  return { label: 'Strong Match', description: 'Aligned with your goals.', tone: 'green' as const };
};

const VERDICT_META = {
  ACCEPT: { label: 'ACCEPT', description: 'Low risk and strong compatibility.', tone: 'green' as const },
  NEGOTIATE: { label: 'NEGOTIATE', description: 'Some red flags or goal conflicts.', tone: 'amber' as const },
  REJECT: { label: 'REJECT', description: 'High risk or poor compatibility.', tone: 'red' as const },
};

const STAGE_PROGRESS: Record<string, number> = {
  'extract': 15,
  'classify': 45,
  'explain': 75,
  'email': 90,
  'done': 100
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [inputMode, setInputMode] = useState<'file' | 'text'>('file');
  const [contractText, setContractText] = useState('');
  const [contractType, setContractType] = useState(CONTRACT_TYPES[0].id);
  
  // Form State
  const [role, setRole] = useState('');
  const [duration, setDuration] = useState('1 Year');
  const [sideProjects, setSideProjects] = useState(true);
  const [relocation] = useState(false);
  const [compensation, setCompensation] = useState('');
  const [requirements, setRequirements] = useState('');
  const [companyName, setCompanyName] = useState('');
  const [userName, setUserName] = useState('');
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [streamStatus, setStreamStatus] = useState<{ stage: string, message: string } | null>(null);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const [expandedFlags, setExpandedFlags] = useState<number[]>([]);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const reportRef = useRef<HTMLDivElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) setFile(e.target.files[0]);
  };

  const handleDragOver = (e: React.DragEvent) => e.preventDefault();
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  };

  const toggleFlag = (index: number) => {
    setExpandedFlags(prev => prev.includes(index) ? prev.filter(i => i !== index) : [...prev, index]);
  };

  const handleDownloadPDF = async () => {
    if (!reportRef.current || isExporting) return;
    setError(null);
    setIsExporting(true);
    try {
      const exportRoot = reportRef.current;
      const exportWidth = exportRoot.scrollWidth || exportRoot.offsetWidth;
      const exportHeight = exportRoot.scrollHeight || exportRoot.offsetHeight;
      const module = await import('html2pdf.js');
      const html2pdf = (module as { default?: unknown }).default ?? module;
      if (typeof html2pdf !== 'function') throw new Error('PDF export is unavailable.');

      const exportStyles = `
        .pdf-export .pdf-only { display: block !important; }
        .pdf-export .screen-only { display: none !important; }
        .pdf-export [data-export-ignore="true"] { display: none !important; }
        .pdf-export .export-root { padding-bottom: 32px !important; }
        .pdf-export .avoid-break { break-inside: avoid !important; page-break-inside: avoid !important; }
        .pdf-export .pdf-page-break { break-before: page !important; page-break-before: always !important; }
        .pdf-export .pdf-section-title { font-size: 18px; font-weight: 700; margin: 18px 0 8px; }
        .pdf-export .pdf-meta-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
        .pdf-export .pdf-card { border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px; background: #ffffff !important; }
        .pdf-export .pdf-badge { font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; }
        .pdf-export .pdf-score { font-size: 28px; font-weight: 700; margin-top: 6px; color: #0f172a; }
        .pdf-export .pdf-note { font-size: 12px; color: #475569; margin-top: 4px; }
        .pdf-export .pdf-safe-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
        .pdf-export .pdf-safe-card { border: 1px solid #dcfce7; background: #f0fdf4 !important; border-radius: 10px; padding: 10px; }
        .pdf-export .pdf-safe-title { font-size: 12px; font-weight: 700; color: #14532d; margin-bottom: 4px; }
        .pdf-export .pdf-safe-text { font-size: 11px; color: #166534; line-height: 1.5; }
        .pdf-export .negotiation-block { background: #ffffff !important; border-color: #e2e8f0 !important; }
        .pdf-export .negotiation-header { break-after: avoid-page; page-break-after: avoid; }
        .pdf-export .negotiation-body { font-size: 12px; line-height: 1.6; color: #0f172a !important; }
        .pdf-export, .pdf-export * {
          color: #0f172a !important;
          border-color: #e2e8f0 !important;
          box-shadow: none !important;
          text-shadow: none !important;
          filter: none !important;
          background: #ffffff !important;
          background-image: none !important;
        }
        .pdf-export { background-color: #ffffff !important; }
        .pdf-export * { background-color: transparent !important; }
        .pdf-export .bg-white { background-color: #ffffff !important; }
        .pdf-export .bg-slate-50 { background-color: #f8fafc !important; }
        .pdf-export .bg-slate-100 { background-color: #f1f5f9 !important; }
        .pdf-export .bg-blue-50 { background-color: #eff6ff !important; }
        .pdf-export .bg-blue-100 { background-color: #dbeafe !important; }
        .pdf-export .bg-amber-50 { background-color: #fffbeb !important; }
        .pdf-export .bg-red-50 { background-color: #fef2f2 !important; }
        .pdf-export .bg-emerald-100 { background-color: #d1fae5 !important; }
        .pdf-export .text-blue-600 { color: #2563eb !important; }
        .pdf-export .text-emerald-600 { color: #059669 !important; }
        .pdf-export .text-amber-600 { color: #d97706 !important; }
        .pdf-export .text-red-600 { color: #dc2626 !important; }
        .pdf-export .text-red-700 { color: #b91c1c !important; }
        .pdf-export .text-green-500 { color: #22c55e !important; }
        .pdf-export .text-red-500 { color: #ef4444 !important; }
        .pdf-export .text-slate-400 { color: #94a3b8 !important; }
        .pdf-export .text-slate-500 { color: #64748b !important; }
        .pdf-export .text-slate-600 { color: #475569 !important; }
        .pdf-export .text-slate-700 { color: #334155 !important; }
        .pdf-export .text-slate-800 { color: #1e293b !important; }
        .pdf-export .text-slate-900 { color: #0f172a !important; }
        .pdf-export .border-slate-200 { border-color: #e2e8f0 !important; }
        .pdf-export .border-red-200 { border-color: #fecaca !important; }
        .pdf-export .border-blue-100 { border-color: #dbeafe !important; }
        .pdf-export .border-amber-200 { border-color: #fde68a !important; }
        .pdf-export .border-green-100 { border-color: #dcfce7 !important; }
      `;

      const sanitizeUnsupportedColors = (doc: Document) => {
        const view = doc.defaultView;
        if (!view) return;

        doc.querySelectorAll<HTMLElement>('*').forEach((el) => {
          const styles = view.getComputedStyle(el);
          for (let i = 0; i < styles.length; i += 1) {
            const prop = styles[i];
            const value = styles.getPropertyValue(prop);
            if (!value) continue;
            if (!value.includes('oklch') && !value.includes('color-mix')) continue;

            let fallback = 'initial';
            if (prop.includes('background')) fallback = '#ffffff';
            if (prop.includes('color') || prop.includes('stroke') || prop.includes('fill')) fallback = '#0f172a';
            if (prop.includes('border')) fallback = '#e2e8f0';
            if (prop.includes('shadow')) fallback = 'none';
            if (prop.includes('filter')) fallback = 'none';
            if (prop.includes('outline')) fallback = '#e2e8f0';
            if (prop.includes('decoration')) fallback = '#0f172a';

            el.style.setProperty(prop, fallback, 'important');
          }
        });
      };

      const opt = {
        margin: 0.5,
        filename: 'FinePrint_Analysis.pdf',
        image: { type: 'jpeg' as const, quality: 0.98 },
        pagebreak: { mode: ['css', 'legacy'] },
        html2canvas: {
          scale: 2,
          useCORS: true,
          logging: false,
          backgroundColor: '#ffffff',
          scrollX: 0,
          scrollY: 0,
          x: 0,
          y: 0,
          width: exportWidth,
          height: exportHeight,
          windowWidth: exportWidth,
          windowHeight: exportHeight,
          onclone: (doc: Document) => {
            // 1. Add pdf-export class for our explicit hex overrides
            doc.documentElement.classList.add('pdf-export');
            // Ensure a neutral background to avoid oklch tokens
            doc.body.style.backgroundColor = '#ffffff';
            doc.body.style.margin = '0';
            doc.body.style.padding = '0';
            doc.documentElement.style.margin = '0';
            doc.documentElement.style.padding = '0';
            // 2. Inject explicit hex color overrides FIRST
            const style = doc.createElement('style');
            style.textContent = exportStyles;
            doc.head.appendChild(style);
            const exportRoot = doc.querySelector('[data-export-root="true"]') as HTMLElement | null;
            if (exportRoot) {
              exportRoot.style.margin = '0';
              exportRoot.style.padding = '0';
              exportRoot.style.maxWidth = 'none';
              exportRoot.style.width = `${exportWidth}px`;
              exportRoot.style.transform = 'none';
              exportRoot.style.left = '0';
            }
            // Force background/gradient cleanup on common wrappers
            doc.querySelectorAll<HTMLElement>('*').forEach((el) => {
              if (el.style && el.style.backgroundImage) {
                el.style.backgroundImage = 'none';
              }
            });
            // 3. Walk every element and replace any remaining oklch / color-mix values
            //    Do NOT remove stylesheets — they contain our overrides
            sanitizeUnsupportedColors(doc);
          }
        },
        jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' as const }
      };

      await (html2pdf as (options?: unknown) => { set: (value: unknown) => { from: (node: HTMLElement) => { save: () => Promise<void> } } })()
        .set(opt)
        .from(exportRoot)
        .save();
    } catch (err) {
      setError(err instanceof Error ? `PDF export failed: ${err.message}` : 'PDF export failed.');
    } finally {
      document.querySelectorAll('.html2pdf__container, .html2pdf__overlay').forEach((node) => node.remove());
      setIsExporting(false);
    }
  };

  const handleAnalyze = async () => {
    const trimmedText = contractText.trim();

    if (inputMode === 'file' && !file) return setError('Please upload a file.');
    if (inputMode === 'text' && !trimmedText) return setError('Please paste the contract text.');

    setIsAnalyzing(true);
    setError(null);
    setStreamStatus({ stage: 'extract', message: 'Initializing pipeline...' });
    setExpandedFlags([]);
    setResult({
      risk_score: null, compatibility_score: null, verdict: null, verdict_reason: null,
      requirement_breakdown: [], red_flags: [], safe_clauses: [], negotiation_email: null
    });

    const combinedRequirements = `
      Role: ${role || 'Not specified'}
      Expected Duration: ${duration}
      Side Projects Allowed: ${sideProjects ? 'Yes' : 'No'}
      Open to Relocation: ${relocation ? 'Yes' : 'No'}
      Minimum Compensation: ${compensation || 'Not specified'}
      Additional Notes: ${requirements.trim()}
    `.trim();

    const formData = new FormData();
    formData.append('contract_type', contractType);
    if (inputMode === 'file' && file) formData.append('file', file);
    if (inputMode === 'text') formData.append('text', trimmedText);
    formData.append('requirements', combinedRequirements);
    if (companyName.trim()) formData.append('company_name', companyName.trim());
    if (userName.trim()) formData.append('user_name', userName.trim());

    try {
      const response = await fetch(`${API_BASE_URL}/analyze/stream`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      if (!response.body) throw new Error('ReadableStream not supported.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        let boundary = buffer.indexOf('\n\n');
        
        // Robust SSE Parsing logic
        while (boundary !== -1) {
          const chunk = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          boundary = buffer.indexOf('\n\n');

          if (!chunk.trim()) continue;
          
          const eventMatch = chunk.match(/event:\s*([^\n]*)/);
          const dataMatch = chunk.match(/data:\s*(.*)/s); // /s flag for multiline json
          
          if (eventMatch && dataMatch) {
            const eventType = eventMatch[1].trim();
            const payload = JSON.parse(dataMatch[1].trim());

            if (eventType === 'status') setStreamStatus(payload);
            else if (eventType === 'error') throw new Error(payload.detail);
            else if (eventType === 'done') {
              setIsAnalyzing(false);
              setStreamStatus({ stage: 'done', message: 'Analysis Complete' });
              setTimeout(() => setStreamStatus(null), 2000);
            }
            else {
              setResult(prev => {
                if (!prev) return prev;
                const next = { ...prev };
                if (eventType === 'risk_score') next.risk_score = payload.risk_score;
                if (eventType === 'compatibility_score') next.compatibility_score = payload.compatibility_score;
                if (eventType === 'verdict') {
                  next.verdict = payload.verdict;
                  next.verdict_reason = payload.verdict_reason;
                }
                if (eventType === 'requirement_match') next.requirement_breakdown = [...next.requirement_breakdown, payload];
                if (eventType === 'safe_clause') next.safe_clauses = [...next.safe_clauses, payload];
                if (eventType === 'red_flag') next.red_flags = [...next.red_flags, payload];
                if (eventType === 'negotiation_email') next.negotiation_email = payload.email;
                return next;
              });
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error');
      setStreamStatus(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const riskMeta = getRiskMeta(result?.risk_score ?? null);
  const compatibilityMeta = getCompatibilityMeta(result?.compatibility_score ?? null);
  const verdictMeta = result?.verdict ? VERDICT_META[result.verdict] : null;
  const currentProgress = streamStatus ? STAGE_PROGRESS[streamStatus.stage] || 0 : (isAnalyzing ? 10 : 0);
  const exportTimestamp = new Date().toISOString().replace('T', ' ').replace('Z', ' UTC');

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText className="text-blue-600 w-6 h-6" />
            <h1 className="text-xl font-bold text-slate-900 tracking-tight">FinePrint</h1>
          </div>
          <div className="text-sm font-medium text-slate-500">Tarkash Labs x Gemma 4</div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8 space-y-8">
        {!result ? (
          <div className="space-y-6 max-w-2xl mx-auto">
            <div className="text-center space-y-2 mb-8">
              <h2 className="text-3xl font-extrabold text-slate-900">Analyze your contract in seconds</h2>
              <p className="text-slate-500">Upload a photo or PDF to instantly identify hidden traps.</p>
            </div>

            <div className="flex items-center justify-center">
              <div className="inline-flex rounded-lg border border-slate-200 bg-white p-1 shadow-sm">
                <button
                  type="button"
                  onClick={() => { setInputMode('file'); setContractText(''); setError(null); }}
                  className={`px-4 py-2 text-sm font-semibold rounded-md transition ${inputMode === 'file' ? 'bg-blue-600 text-white' : 'text-slate-600'}`}
                >
                  Upload File
                </button>
                <button
                  type="button"
                  onClick={() => { setInputMode('text'); setFile(null); setError(null); }}
                  className={`px-4 py-2 text-sm font-semibold rounded-md transition ${inputMode === 'text' ? 'bg-blue-600 text-white' : 'text-slate-600'}`}
                >
                  Paste Text
                </button>
              </div>
            </div>

            {inputMode === 'file' ? (
              <div 
                onDragOver={handleDragOver} onDrop={handleDrop} onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-300 rounded-2xl bg-white p-10 flex flex-col items-center justify-center gap-4 hover:bg-slate-50 cursor-pointer"
              >
                <input type="file" ref={fileInputRef} className="hidden" accept="image/*,application/pdf" onChange={handleFileChange} />
                <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center text-blue-600 mb-2">
                  <Upload className="w-8 h-8" />
                </div>
                <div className="text-center">
                  <p className="text-lg font-medium text-slate-900">{file ? file.name : "Click or drag to upload"}</p>
                </div>
              </div>
            ) : (
              <div className="border border-slate-200 rounded-2xl bg-white p-6 shadow-sm">
                <label className="block text-sm font-semibold text-slate-700 mb-2">Paste Contract Text</label>
                <textarea
                  value={contractText} onChange={(e) => setContractText(e.target.value)} rows={8}
                  className="w-full resize-y rounded-lg border border-slate-300 px-4 py-3 text-sm text-slate-900 outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            )}

            <div className="space-y-3">
              <label className="block text-sm font-semibold text-slate-700">Contract Type</label>
              <select value={contractType} onChange={(e) => setContractType(e.target.value)} className="w-full bg-white border border-slate-300 rounded-lg px-4 py-3 text-slate-900 outline-none focus:ring-2 focus:ring-blue-500">
                {CONTRACT_TYPES.map(type => <option key={type.id} value={type.id}>{type.label}</option>)}
              </select>
            </div>

            {/* Structured Hybrid Form */}
            <div className="border border-slate-200 rounded-2xl bg-white p-6 shadow-sm space-y-4">
              <h3 className="text-sm font-bold text-slate-800 border-b pb-2">Your Guardrails & Requirements</h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-600 mb-1">Target Role</label>
                  <input type="text" value={role} onChange={e => setRole(e.target.value)} placeholder="e.g. SDE, Intern" className="w-full border rounded-md px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-blue-500"/>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-600 mb-1">Min. Comp/Stipend</label>
                  <input type="text" value={compensation} onChange={e => setCompensation(e.target.value)} placeholder="e.g. ₹50k/mo" className="w-full border rounded-md px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-blue-500"/>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-600 mb-1">Max Duration / Bond</label>
                  <select value={duration} onChange={e => setDuration(e.target.value)} className="w-full border rounded-md px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-blue-500">
                    <option>&lt; 6 Months</option>
                    <option>1 Year</option>
                    <option>2 Years</option>
                    <option>No Limit</option>
                  </select>
                </div>
                <div className="flex items-center justify-between px-2 pt-5">
                  <label className="text-xs font-semibold text-slate-600">Need Side Projects?</label>
                  <input type="checkbox" checked={sideProjects} onChange={e => setSideProjects(e.target.checked)} className="w-4 h-4 rounded text-blue-600"/>
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-600 mb-1">Additional Needs (Free Text)</label>
                <textarea
                  value={requirements} onChange={(e) => setRequirements(e.target.value)} rows={2}
                  placeholder="Example: Need flexible hours for exams..."
                  className="w-full resize-y rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>
            </div>

            {/* Email Generator Context */}
            <div className="border border-slate-200 rounded-2xl bg-white p-6 shadow-sm space-y-4">
              <h3 className="text-sm font-bold text-slate-800 border-b pb-2">Negotiation Details</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="relative">
                  <Building className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
                  <input type="text" value={companyName} onChange={e => setCompanyName(e.target.value)} placeholder="Company Name" className="w-full pl-9 pr-3 py-2 border rounded-md text-sm outline-none focus:ring-1 focus:ring-blue-500"/>
                </div>
                <div className="relative">
                  <User className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
                  <input type="text" value={userName} onChange={e => setUserName(e.target.value)} placeholder="Your Full Name" className="w-full pl-9 pr-3 py-2 border rounded-md text-sm outline-none focus:ring-1 focus:ring-blue-500"/>
                </div>
              </div>
            </div>

            <button 
              onClick={handleAnalyze}
              disabled={isAnalyzing || (inputMode === 'file' ? !file : contractText.trim().length === 0)}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-semibold rounded-lg px-4 py-3 shadow-sm transition-all"
            >
              {isAnalyzing ? 'Analyzing Pipeline...' : 'Analyze Contract'}
            </button>

            {/* Streaming Progress Bar */}
            {(isAnalyzing || streamStatus?.stage === 'done') && (
              <div className="w-full space-y-2 mt-4 animate-fade-in">
                <div className="flex justify-between text-xs font-bold text-slate-500 uppercase">
                  <span>Progress</span>
                  <span className="text-blue-600">{streamStatus?.message || 'Processing...'}</span>
                </div>
                <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-blue-500 transition-all duration-500 ease-out" 
                    style={{ width: `${currentProgress}%` }}
                  />
                </div>
              </div>
            )}

            {error && <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>}
          </div>
        ) : (
          <div className="space-y-8 export-root" ref={reportRef} data-export-root="true">
            <div className="hidden pdf-only">
              <div className="flex items-start justify-between border-b border-slate-200 pb-4">
                <div>
                  <h2 className="text-2xl font-bold">FinePrint Analysis Report</h2>
                  <p className="text-sm text-slate-500">Generated {exportTimestamp}</p>
                </div>
                <div className="text-right text-xs text-slate-500">
                  <div className="font-semibold text-slate-700">Contract Type</div>
                  <div>{CONTRACT_TYPES.find(c => c.id === contractType)?.label || 'General Contract'}</div>
                </div>
              </div>
              <div className="pdf-section-title">Executive Summary</div>
              <div className="pdf-meta-grid">
                <div className="pdf-card">
                  <div className="pdf-badge">Risk Score</div>
                  <div className="pdf-score">{result.risk_score !== null ? result.risk_score : '--'}</div>
                  <div className="pdf-note">{riskMeta.label} • {riskMeta.description}</div>
                </div>
                <div className="pdf-card">
                  <div className="pdf-badge">Compatibility</div>
                  <div className="pdf-score">{result.compatibility_score !== null ? result.compatibility_score : '--'}</div>
                  <div className="pdf-note">{compatibilityMeta.label} • {compatibilityMeta.description}</div>
                </div>
                <div className="pdf-card">
                  <div className="pdf-badge">Final Verdict</div>
                  <div className="pdf-score">{result.verdict ?? '--'}</div>
                  <div className="pdf-note">{result.verdict_reason || 'Verdict pending.'}</div>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between border-b pb-4 screen-only" data-export-ignore="true">
              <div>
                <h2 className="text-2xl font-bold flex items-center gap-3">
                  Analysis Results
                  <span className="text-xs bg-slate-200 text-slate-600 px-2 py-1 rounded-full uppercase tracking-wider">
                    {CONTRACT_TYPES.find(c => c.id === contractType)?.label}
                  </span>
                </h2>
                {isAnalyzing && streamStatus && (
                  <p className="text-sm text-blue-600 mt-2 flex items-center gap-2 font-medium">
                    <Loader2 className="w-4 h-4 animate-spin" /> {streamStatus.message}
                  </p>
                )}
              </div>
              <div className="flex gap-3">
                <button onClick={handleDownloadPDF} disabled={isAnalyzing || isExporting} className="text-sm font-semibold bg-slate-900 hover:bg-slate-800 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition disabled:opacity-50">
                  {isExporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                  {isExporting ? 'Exporting…' : 'Export PDF'}
                </button>
                <button onClick={() => { setResult(null); setFile(null); setError(null); setContractText(''); }} className="text-sm font-medium bg-blue-50 hover:bg-blue-100 text-blue-700 px-4 py-2 rounded-lg transition">
                  New Analysis
                </button>
              </div>
            </div>

            {error && (
              <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            )}

            <div className="grid gap-4 md:grid-cols-3 screen-only">
              {/* Risk Score */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">Risk Score</h3>
                    <span className={`text-xs font-semibold px-2 py-1 rounded-full ${SCORE_TONE_STYLES[riskMeta.tone].badge}`}>{riskMeta.label}</span>
                  </div>
                  <div className="mt-4 flex items-baseline gap-2">
                    <span className={`text-5xl font-extrabold ${SCORE_TONE_STYLES[riskMeta.tone].text}`}>
                      {result.risk_score !== null ? result.risk_score : '--'}
                    </span>
                    <span className="text-sm text-slate-500">/ 100</span>
                  </div>
                </div>
                <div>
                  <p className="mt-4 text-sm text-slate-600">{riskMeta.description}</p>
                  {result.risk_score !== null && (
                    <p className="mt-1 text-[11px] text-slate-400 font-medium">
                      Higher risk than {Math.min(99, Math.max(1, result.risk_score))}% of {CONTRACT_TYPES.find(c => c.id === contractType)?.label.toLowerCase() || 'contracts'}.
                    </p>
                  )}
                </div>
              </div>

              {/* Compatibility Score & Checklist */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">Compatibility</h3>
                  <span className={`text-xs font-semibold px-2 py-1 rounded-full ${SCORE_TONE_STYLES[compatibilityMeta.tone].badge}`}>{compatibilityMeta.label}</span>
                </div>
                <div className="mt-4 flex items-baseline gap-2">
                  <span className={`text-5xl font-extrabold ${SCORE_TONE_STYLES[compatibilityMeta.tone].text}`}>
                    {result.compatibility_score !== null ? result.compatibility_score : '--'}
                  </span>
                  <span className="text-sm text-slate-500">/ 100</span>
                </div>
                <div className="mt-4 space-y-2 flex-grow border-t pt-3">
                  {result.requirement_breakdown.length > 0 ? (
                    result.requirement_breakdown.map((req, i) => (
                      <div key={i} className="flex items-start gap-2 text-sm">
                        {req.met ? <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 shrink-0" /> : <XCircle className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />}
                        <span className={req.met ? "text-slate-700" : "text-red-700 font-medium"}>{req.requirement}</span>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-slate-400 italic">No specific requirements analyzed.</p>
                  )}
                </div>
              </div>

              {/* Verdict */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">Final Verdict</h3>
                  {verdictMeta && <span className={`text-xs font-semibold px-2 py-1 rounded-full ${SCORE_TONE_STYLES[verdictMeta.tone].badge}`}>{verdictMeta.label}</span>}
                </div>
                <div className="mt-4 flex-grow">
                  <p className="text-sm font-semibold text-slate-900 leading-relaxed text-lg">
                    {result.verdict_reason || (isAnalyzing ? "Computing verdict..." : "")}
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="font-bold flex items-center gap-2 text-slate-800 border-b pb-2">
                <AlertTriangle className="w-5 h-5 text-red-500" />
                Red Flags ({result.red_flags.length})
                {isAnalyzing && <span className="ml-2 text-xs bg-amber-100 text-amber-700 px-2 py-1 rounded animate-pulse">Extracting via MoE...</span>}
              </h3>
              
              <div className="grid gap-4">
                {result.red_flags.map((flag, i) => {
                  const isExpanded = expandedFlags.includes(i);
                  return (
                    <div key={i} className="bg-white border border-red-200 rounded-xl overflow-hidden shadow-sm transition-all hover:border-red-300">
                      <div 
                        className="p-5 flex justify-between items-start cursor-pointer bg-red-50/50"
                        onClick={() => toggleFlag(i)}
                      >
                        <div className="flex-grow pr-4">
                          <div className="flex items-center gap-3 mb-2">
                            <h4 className="font-semibold text-red-900 text-lg">{flag.clause_title}</h4>
                            <span className="bg-red-100 text-red-700 text-[10px] font-bold px-2 py-0.5 rounded uppercase">{flag.severity}</span>
                          </div>
                          <p className="text-slate-700 text-sm leading-relaxed">{flag.plain_english_explanation}</p>
                        </div>
                        <button className="text-red-400 hover:text-red-600 p-1">
                          {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                        </button>
                      </div>
                      
                      {isExpanded && (
                        <div className="p-5 border-t border-red-100 bg-white space-y-4">
                          <div>
                            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1 block">Original Contract Text</span>
                            <div className="bg-slate-50 border border-slate-200 p-3 rounded-lg text-sm font-mono text-slate-600 whitespace-pre-wrap">
                              "{flag.clause_text}"
                            </div>
                          </div>
                          <div className="bg-amber-50 border border-amber-200 p-4 rounded-lg flex items-start gap-3">
                            <div className="text-amber-800 text-sm">
                              <span className="font-bold uppercase text-xs tracking-wider block mb-1">💡 How to Negotiate</span>
                              {flag.negotiation_tip}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
                {!isAnalyzing && result.red_flags.length === 0 && (
                  <div className="text-center py-8 border-2 border-dashed border-slate-200 rounded-xl text-slate-500">
                    No major red flags found.
                  </div>
                )}
              </div>
            </div>

            {/* Email Generator UI */}
            {(!isAnalyzing || result.negotiation_email) && result.red_flags.length > 0 && (
                <div className="space-y-4 mt-8 bg-blue-50/50 p-6 rounded-2xl border border-blue-100 negotiation-block">
                  <div className="flex items-center justify-between border-b border-blue-200 pb-2 negotiation-header">
                    <h3 className="font-bold flex items-center gap-2 text-blue-900">
                      Draft Negotiation Email
                      <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">Gemma 27B</span>
                    </h3>
                    {result.negotiation_email && (
                      <button 
                        onClick={() => navigator.clipboard.writeText(result.negotiation_email || '')}
                        className="text-sm font-medium text-blue-600 hover:text-blue-800 flex items-center gap-1"
                        data-export-ignore="true"
                      >
                        <Copy className="w-4 h-4" /> Copy Text
                      </button>
                    )}
                 </div>
                 {result.negotiation_email ? (
                   <div className="bg-white border border-blue-200 rounded-lg p-5 text-sm text-slate-700 whitespace-pre-wrap font-serif leading-relaxed shadow-inner negotiation-body">
                     {result.negotiation_email}
                   </div>
                 ) : (
                   <p className="text-sm text-slate-500 italic">Email draft will appear here...</p>
                 )}
              </div>
            )}

            <div className="hidden pdf-only pdf-page-break">
              <div className="pdf-section-title">Safe Clauses</div>
              {result.safe_clauses.length > 0 ? (
                <div className="pdf-safe-grid">
                  {result.safe_clauses.map((clause, i) => (
                    <div key={`pdf-safe-${i}`} className="pdf-safe-card">
                      <div className="pdf-safe-title">{clause.clause_title}</div>
                      <div className="pdf-safe-text">{clause.plain_english_explanation}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="pdf-note">No safe clauses found for this contract.</div>
              )}
            </div>

            <div className="space-y-4 pt-4 pdf-page-break screen-only">
              <h3 className="font-bold flex items-center gap-2 text-slate-800 border-b pb-2">
                <ShieldCheck className="w-5 h-5 text-green-500" />
                Safe Clauses ({result.safe_clauses.length})
              </h3>
              <div className="grid gap-4 md:grid-cols-2">
                {result.safe_clauses.map((clause, i) => (
                  <div key={i} className="bg-green-50/50 border border-green-100 rounded-xl p-4 avoid-break">
                    <h4 className="font-semibold text-green-900 mb-1 text-sm">{clause.clause_title}</h4>
                    <p className="text-green-800/80 text-xs leading-relaxed">{clause.plain_english_explanation}</p>
                  </div>
                ))}
                {!isAnalyzing && result.safe_clauses.length === 0 && (
                  <div className="col-span-full text-center py-6 border-2 border-dashed border-red-200 bg-red-50 rounded-xl text-red-600 font-medium">
                    No safe clauses found. This contract is heavily one-sided.
                  </div>
                )}
              </div>
            </div>

          </div>
        )}
      </main>
    </div>
  );
}

export default App;