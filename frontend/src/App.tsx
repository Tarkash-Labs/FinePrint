import { useEffect, useState, useRef } from 'react';
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

type FieldConfig = { key: string; label: string; type: 'text' | 'select' | 'checkbox'; options?: string[]; placeholder?: string; };

const REQUIREMENT_FIELDS: Record<string, FieldConfig[]> = {
  employment: [
    { key: 'role', label: 'Target Role', type: 'text', placeholder: 'e.g. SDE, Manager' },
    { key: 'compensation', label: 'Min. Compensation', type: 'text', placeholder: 'e.g. ₹50k/mo' },
    { key: 'duration', label: 'Max Duration / Bond', type: 'select', options: ['< 6 Months', '1 Year', '2 Years', 'No Limit'] },
    { key: 'sideProjects', label: 'Need Side Projects?', type: 'checkbox' },
    { key: 'relocation', label: 'Open to Relocation?', type: 'checkbox' }
  ],
  internship: [
    { key: 'role', label: 'Target Role', type: 'text', placeholder: 'e.g. SDE Intern' },
    { key: 'compensation', label: 'Min. Stipend', type: 'text', placeholder: 'e.g. ₹20k/mo' },
    { key: 'duration', label: 'Max Duration', type: 'select', options: ['< 6 Months', '1 Year', 'No Limit'] },
    { key: 'sideProjects', label: 'Need Side Projects?', type: 'checkbox' }
  ],
  freelance: [
    { key: 'scope', label: 'Project Scope', type: 'text', placeholder: 'e.g. Web App UI' },
    { key: 'compensation', label: 'Min. Payment', type: 'text', placeholder: 'e.g. ₹1L total' },
    { key: 'duration', label: 'Expected Duration', type: 'select', options: ['< 1 Month', '< 3 Months', '6 Months', 'Ongoing'] },
    { key: 'exclusivity', label: 'Agree to Exclusivity?', type: 'checkbox' }
  ],
  rental: [
    { key: 'property', label: 'Property Type', type: 'text', placeholder: 'e.g. 2BHK Apartment' },
    { key: 'deposit', label: 'Max Security Deposit', type: 'text', placeholder: 'e.g. 2 Months Rent' },
    { key: 'duration', label: 'Lease Duration', type: 'select', options: ['6 Months', '11 Months', '1 Year', '2 Years+'] },
    { key: 'pets', label: 'Pets Allowed?', type: 'checkbox' }
  ],
  vc: [
    { key: 'valuation', label: 'Target Valuation', type: 'text', placeholder: 'e.g. $10M Post-money' },
    { key: 'investment', label: 'Target Investment', type: 'text', placeholder: 'e.g. $2M' },
    { key: 'boardSeats', label: 'Give up Board Seat?', type: 'checkbox' }
  ],
  tos: [
    { key: 'usage', label: 'Primary Usage', type: 'text', placeholder: 'e.g. Personal, Business' },
  ],
  general: [
    { key: 'goal', label: 'Primary Goal', type: 'text', placeholder: 'e.g. Partnership' },
  ]
};

type Severity = 'high' | 'medium' | 'low';
type Verdict = 'ACCEPT' | 'NEGOTIATE' | 'REJECT';

type RequirementMatch = { requirement: string; met: boolean; explanation: string; };
type FlaggedClause = { clause_title: string; clause_text: string; plain_english_explanation: string; negotiation_tip: string; suggested_rewrite?: string; severity: Severity; };
type SafeClause = { clause_title: string; plain_english_explanation: string; };
type ChatMessage = { role: 'user' | 'assistant'; content: string; };
type ClauseChatState = { messages: ChatMessage[]; input: string; isLoading: boolean; error?: string | null; };
type Timing = { e4b_ms?: number; moe_ms?: number; dense_ms?: number; };

type AnalyzeResult = {
  risk_score: number | null;
  compatibility_score: number | null;
  verdict: Verdict | null;
  verdict_reason: string | null;
  summary: string | null;
  requirement_breakdown: RequirementMatch[];
  red_flags: FlaggedClause[];
  safe_clauses: SafeClause[];
  negotiation_email: string | null;
  timing?: Timing | null;
};

// NEW: Type for the Comparison Endpoint
type CompareResult = {
  summary: string;
  resolved_flags: string[];
  new_flags: string[];
  remaining_flags: string[];
  overall_change: 'IMPROVED' | 'WORSE' | 'UNCHANGED';
};

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const SCORE_TONE_STYLES = {
  green: { badge: 'bg-emerald-100 text-emerald-700', text: 'text-emerald-600' },
  amber: { badge: 'bg-amber-100 text-amber-700', text: 'text-amber-600' },
  red: { badge: 'bg-red-100 text-red-700', text: 'text-red-600' },
  gray: { badge: 'bg-slate-100 text-slate-700', text: 'text-slate-400' },
} as const;

const RISK_BENCHMARKS: Record<string, { min: number; max: number }> = {
  employment: { min: 40, max: 65 },
  internship: { min: 35, max: 60 },
  rental: { min: 30, max: 55 },
  freelance: { min: 45, max: 70 },
  vc: { min: 55, max: 80 },
  tos: { min: 50, max: 75 },
  general: { min: 40, max: 65 },
};

const formatMs = (value?: number) => (typeof value === 'number' ? `${(value / 1000).toFixed(1)}s` : '--');

const getBenchmarkNote = (score: number | null, contractId: string) => {
  if (score === null) return null;
  const benchmark = RISK_BENCHMARKS[contractId] || RISK_BENCHMARKS.general;
  const label = CONTRACT_TYPES.find(type => type.id === contractId)?.label || 'contracts';
  if (score <= benchmark.min) return `Better than most typical ${label.toLowerCase()} contracts.`;
  if (score >= benchmark.max) return `Worse than 90%+ of typical ${label.toLowerCase()} contracts.`;
  const percentile = Math.round(((score - benchmark.min) / (benchmark.max - benchmark.min)) * 100);
  return `Worse than about ${percentile}% of typical ${label.toLowerCase()} contracts.`;
};

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

const ANALYSIS_STEPS = [
  { key: 'extract', label: 'Extract' },
  { key: 'analyze', label: 'Analyze' },
  { key: 'explain', label: 'Explain' },
  { key: 'email', label: 'Email' },
];

const EXAMPLES = [
  { label: '🎓 Campus Bond', type: 'internship', text: `1. Position: Software Engineering Intern at Initech Solutions.\n2. Compensation: This is an unpaid position. No stipend will be provided.\n3. Intellectual Property: All software and code created during the term, including on personal time using personal equipment, shall become exclusive property of Initech Solutions.\n4. Bond: If hired full-time, intern must stay 3 years or pay ₹10,000,000 penalty.\n5. Working Hours: Standard hours 10 AM–7 PM. Nights and weekends expected during deadlines.\n6. Non-Compete: Cannot join any competitor for 18 months after leaving.` },
  { label: '📝 Predatory NDA', type: 'freelance', text: `1. IP Assignment: All work product, including work on personal time using personal tools, is owned exclusively by Client.\n2. Non-Solicitation: Freelancer may not work with Client's clients or competitors for 24 months.\n3. Payment: Client may withhold payment if deliverables deemed unsatisfactory at Client's sole discretion.\n4. Exclusivity: No other clients permitted during project without written approval.\n5. Termination: Client may terminate with no notice. Freelancer must give 60 days notice.` },
  { label: '🏠 Shady Rental', type: 'rental', text: `1. Security Deposit: 6 months rent due before move-in. Refundable at landlord's sole discretion.\n2. Landlord Access: Landlord may enter at any time without notice.\n3. Repairs: Tenant responsible for all repairs under ₹10,000.\n4. Eviction: Landlord may terminate with 7 days notice for any reason.\n5. Auto-Renewal: Agreement auto-renews for 11 months unless tenant gives 60 days written notice.` },
];

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [inputMode, setInputMode] = useState<'file' | 'text'>('file');
  const [contractText, setContractText] = useState('');
  const [contractType, setContractType] = useState(CONTRACT_TYPES[0].id);
  
  // NEW: Compare Mode State
  const [isCompareMode, setIsCompareMode] = useState(false);
  const [fileV2, setFileV2] = useState<File | null>(null);
  const [contractTextV2, setContractTextV2] = useState('');
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null);
  const [isComparing, setIsComparing] = useState(false);

  // Dynamic Form State
  const [dynamicReqs, setDynamicReqs] = useState<Record<string, any>>({
    sideProjects: true,
    relocation: false,
    exclusivity: false,
    pets: false,
    boardSeats: false,
  });
  
  // Universal Form State
  const [requirements, setRequirements] = useState('');
  const [companyName, setCompanyName] = useState('');
  const [userName, setUserName] = useState('');
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [streamStatus, setStreamStatus] = useState<{ stage: string, message: string } | null>(null);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [shareUrl, setShareUrl] = useState<string | null>(null);
  const [isLoadingReport, setIsLoadingReport] = useState(false);
  const [clauseChats, setClauseChats] = useState<Record<number, ClauseChatState>>({});
  
  const [expandedFlags, setExpandedFlags] = useState<number[]>([]);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const fileInputRefV2 = useRef<HTMLInputElement>(null); // NEW: V2 Ref
  const reportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const reportId = params.get('report');
    if (!reportId) return;

    setIsLoadingReport(true);
    setError(null);

    fetch(`${API_BASE_URL}/report/${reportId}`)
      .then(async (res) => {
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `Request failed (${res.status})`);
        }
        return res.json() as Promise<{ contract_type: string; analysis: AnalyzeResult }>;
      })
      .then((payload) => {
        setContractType(payload.contract_type || CONTRACT_TYPES[0].id);
        setResult(payload.analysis);
        setShareUrl(window.location.href);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : 'Failed to load shared report.');
      })
      .finally(() => setIsLoadingReport(false));
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) setFile(e.target.files[0]);
  };
  
  // NEW: Handlers for V2 File
  const handleFileChangeV2 = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) setFileV2(e.target.files[0]);
  };

  const handleDragOver = (e: React.DragEvent) => e.preventDefault();
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  };
  const handleDropV2 = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) setFileV2(e.dataTransfer.files[0]);
  };

  const toggleFlag = (index: number) => {
    setExpandedFlags(prev => prev.includes(index) ? prev.filter(i => i !== index) : [...prev, index]);
  };

  const updateClauseChat = (index: number, next: Partial<ClauseChatState>) => {
    setClauseChats(prev => {
      const current = prev[index] || { messages: [], input: '', isLoading: false, error: null };
      return { ...prev, [index]: { ...current, ...next } };
    });
  };

  const handleAskClause = async (index: number) => {
    if (!result) return;
    const chat = clauseChats[index] || { messages: [], input: '', isLoading: false, error: null };
    const question = chat.input.trim();
    if (!question) return;

    const flag = result.red_flags[index];
    const nextMessages = [...chat.messages, { role: 'user', content: question } as ChatMessage];
    updateClauseChat(index, { messages: nextMessages, input: '', isLoading: true, error: null });

    try {
      const response = await fetch(`${API_BASE_URL}/ask-clause`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contract_type: contractType,
          clause_title: flag.clause_title,
          clause_text: flag.clause_text,
          question,
          history: chat.messages,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Request failed (${response.status})`);
      }

      const payload = await response.json() as { answer?: string };
      const answer = payload.answer || 'No response available.';
      updateClauseChat(index, {
        messages: [...nextMessages, { role: 'assistant', content: answer }],
        isLoading: false,
      });
    } catch (err) {
      updateClauseChat(index, {
        isLoading: false,
        error: err instanceof Error ? err.message : 'Failed to get a response.',
      });
    }
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
            doc.documentElement.classList.add('pdf-export');
            doc.body.style.backgroundColor = '#ffffff';
            doc.body.style.margin = '0';
            doc.body.style.padding = '0';
            doc.documentElement.style.margin = '0';
            doc.documentElement.style.padding = '0';
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
            doc.querySelectorAll<HTMLElement>('*').forEach((el) => {
              if (el.style && el.style.backgroundImage) {
                el.style.backgroundImage = 'none';
              }
            });
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

  const activeFields = REQUIREMENT_FIELDS[contractType] || REQUIREMENT_FIELDS.general;

  // Build the shared dynamic requirements payload
  const buildRequirementsText = () => {
    const dynamicFieldsText = activeFields.map(f => {
      const rawVal = dynamicReqs[f.key];
      const val = rawVal !== undefined ? rawVal : (f.type === 'select' ? f.options?.[0] : (f.type === 'checkbox' ? false : ''));
      const displayVal = f.type === 'checkbox' ? (val ? 'Yes' : 'No') : (val || 'Not specified');
      return `${f.label}: ${displayVal}`;
    }).join('\n      ');

    return `
      ${dynamicFieldsText}
      Additional Notes: ${requirements.trim()}
    `.trim();
  };

  // NEW: Handler for Comparison Request
  const handleCompare = async () => {
    const trimmedText1 = contractText.trim();
    const trimmedText2 = contractTextV2.trim();

    if (inputMode === 'file' && (!file || !fileV2)) return setError('Please upload both original and revised files to compare.');
    if (inputMode === 'text' && (!trimmedText1 || !trimmedText2)) return setError('Please paste both contract texts to compare.');

    setIsComparing(true);
    setError(null);
    setCompareResult(null);
    setResult(null); // Clear standard result

    const formData = new FormData();
    formData.append('contract_type', contractType);
    formData.append('requirements', buildRequirementsText());
    
    if (inputMode === 'file') {
      if (file) formData.append('file_v1', file);
      if (fileV2) formData.append('file_v2', fileV2);
    } else {
      formData.append('text_v1', trimmedText1);
      formData.append('text_v2', trimmedText2);
    }

    try {
      const response = await fetch(`${API_BASE_URL}/compare`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || `Comparison failed (${response.status})`);
      }

      const data = await response.json();
      setCompareResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Comparison failed.');
    } finally {
      setIsComparing(false);
    }
  };

  const handleAnalyze = async () => {
    const trimmedText = contractText.trim();

    if (inputMode === 'file' && !file) return setError('Please upload a file.');
    if (inputMode === 'text' && !trimmedText) return setError('Please paste the contract text.');

    setIsAnalyzing(true);
    setError(null);
    setStreamStatus({ stage: 'extract', message: 'Extracting contract text...' });
    setExpandedFlags([]);
    setCompareResult(null);
    setResult({
      risk_score: null, compatibility_score: null, verdict: null, verdict_reason: null,
      summary: null, requirement_breakdown: [], red_flags: [], safe_clauses: [], negotiation_email: null, timing: null
    });

    const formData = new FormData();
    formData.append('contract_type', contractType);
    if (inputMode === 'file' && file) formData.append('file', file);
    if (inputMode === 'text') formData.append('text', trimmedText);
    formData.append('requirements', buildRequirementsText());
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
        
        while (boundary !== -1) {
          const chunk = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          boundary = buffer.indexOf('\n\n');

          if (!chunk.trim()) continue;
          
          const eventMatch = chunk.match(/event:\s*([^\n]*)/);
          const dataMatch = chunk.match(/data:\s*(.*)/s); 
          
          if (eventMatch && dataMatch) {
            const eventType = eventMatch[1].trim();
            const payload = JSON.parse(dataMatch[1].trim());

            if (eventType === 'status') setStreamStatus(payload);
            else if (eventType === 'error') throw new Error(payload.detail);
            else if (eventType === 'done') {
              setResult(prev => prev ? { ...prev, timing: payload.timing || prev.timing } : prev);
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
                if (eventType === 'summary') next.summary = payload.summary;
                if (eventType === 'requirement_match') next.requirement_breakdown = [...next.requirement_breakdown, payload];
                if (eventType === 'safe_clause') next.safe_clauses = [...next.safe_clauses, payload];
                if (eventType === 'red_flag') next.red_flags = [...next.red_flags, payload];
                if (eventType === 'negotiation_email') next.negotiation_email = payload.email;
                return next;
              });
            }
            if (eventType === 'share_report') {
              const base = `${window.location.origin}${window.location.pathname}`;
              setShareUrl(`${base}?report=${payload.report_id}`);
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
  const benchmarkNote = getBenchmarkNote(result?.risk_score ?? null, contractType);
  const riskPercent = typeof result?.risk_score === 'number'
    ? Math.min(Math.max(result.risk_score, 0), 100)
    : 0;
  const compatibilityPercent = typeof result?.compatibility_score === 'number'
    ? Math.min(Math.max(result.compatibility_score, 0), 100)
    : 0;
  const verdictToneClass = verdictMeta ? `verdict-${verdictMeta.tone}` : 'verdict-neutral';
  const activeStepIndex = streamStatus
    ? (streamStatus.stage === 'done'
        ? ANALYSIS_STEPS.length
        : ANALYSIS_STEPS.findIndex(step => step.key === streamStatus.stage))
    : -1;
  const exportTimestamp = new Date().toISOString().replace('T', ' ').replace('Z', ' UTC');

  // Disable submit button based on mode
  const isSubmitDisabled = isCompareMode
      ? (isComparing || (inputMode === 'file' ? (!file || !fileV2) : (!contractText.trim() || !contractTextV2.trim())))
      : (isAnalyzing || (inputMode === 'file' ? !file : contractText.trim().length === 0));

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="container flex items-center justify-between py-4">
          <div className="brand">
            <div className="brand-mark">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <div className="text-lg">FinePrint</div>
              <div className="text-xs text-slate-500">AI contract guardian</div>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-600">
            <span className="pill">Gemma 4</span>
            <span className="hidden sm:inline">Tarkash Labs</span>
          </div>
        </div>
      </header>

      <main className="app-main container">
        {/* Conditional rendering based on whether we have results to show */}
        {!result && !compareResult ? (
          <div className="landing space-y-10 reveal">
            {isLoadingReport ? (
              <div className="card text-sm text-slate-600">
                Loading shared report...
              </div>
            ) : null}

            <section className="hero-grid reveal">
              <div className="space-y-6">
                <span className="badge badge-glow">Contract intelligence</span>
                <h2 className="hero-title">
                  Spot hidden traps <span className="hero-accent">before you sign.</span>
                </h2>
                <p className="hero-subtitle">
                  FinePrint reads contracts like a senior counsel and explains the risk in clear,
                  actionable language. Upload a PDF or paste the text. Gemma does the rest.
                </p>

                <div className="hero-stats">
                  <div className="stat-card">
                    <div className="stat-value">4</div>
                    <div className="stat-label">analysis stages</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">3</div>
                    <div className="stat-label">actionable outputs</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">1</div>
                    <div className="stat-label">shareable report</div>
                  </div>
                </div>

                <div className="metric-grid">
                  <div className="metric-card">
                    <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Signals</div>
                    <div className="mt-2 text-lg font-semibold">Red flags ranked by severity</div>
                  </div>
                  <div className="metric-card">
                    <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Clarity</div>
                    <div className="mt-2 text-lg font-semibold">Plain English, clause by clause</div>
                  </div>
                  <div className="metric-card">
                    <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Action</div>
                    <div className="mt-2 text-lg font-semibold">Negotiation-ready rewrites</div>
                  </div>
                </div>

                <div className="hero-preview">
                  <div className="preview-header">
                    <span className="section-title">Sample verdict</span>
                    <span className="verdict-chip verdict-amber">NEGOTIATE</span>
                  </div>
                  <div className="preview-scores">
                    <div>
                      <div className="preview-label">Risk</div>
                      <div className="preview-value">68</div>
                    </div>
                    <div>
                      <div className="preview-label">Fit</div>
                      <div className="preview-value">52</div>
                    </div>
                    <div>
                      <div className="preview-label">Flags</div>
                      <div className="preview-value">5</div>
                    </div>
                  </div>
                  <p className="preview-note">Top issue: non-compete scope spans 18 months.</p>
                </div>

                <div className="flex flex-wrap gap-2">
                  <span className="pill"><ShieldCheck className="w-4 h-4" /> Auto summaries</span>
                  <span className="pill"><AlertTriangle className="w-4 h-4" /> Risk scoring</span>
                  <span className="pill"><FileText className="w-4 h-4" /> Shareable report</span>
                </div>
              </div>

              <div className="card form-panel space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="section-title">Start analysis</div>
                    <p className="text-sm text-slate-500 mt-2">Upload or paste your contract to begin.</p>
                  </div>
                  
                  {/* NEW: Compare Mode Toggle Switch */}
                  <label className="flex flex-col items-end gap-1 cursor-pointer group">
                    <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Compare Mode</span>
                    <div className="relative">
                      <input type="checkbox" className="sr-only" checked={isCompareMode} onChange={e => setIsCompareMode(e.target.checked)} />
                      <div className={`block w-10 h-6 rounded-full transition ${isCompareMode ? 'bg-teal-600' : 'bg-slate-300'}`}></div>
                      <div className={`absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition transform ${isCompareMode ? 'translate-x-4' : ''}`}></div>
                    </div>
                  </label>
                </div>

                <div className="form-steps">
                  <div className="form-step is-active"><span>01</span> Upload</div>
                  <div className="form-step"><span>02</span> Preferences</div>
                  <div className="form-step"><span>03</span> Analyze</div>
                </div>

                <div className="flex items-center justify-center">
                  <div className="tab-switch">
                    <button
                      type="button"
                      onClick={() => { setInputMode('file'); setContractText(''); setContractTextV2(''); setError(null); }}
                      className={`tab-button ${inputMode === 'file' ? 'is-active' : ''}`}
                    >
                      Upload File
                    </button>
                    <button
                      type="button"
                      onClick={() => { setInputMode('text'); setFile(null); setFileV2(null); setError(null); }}
                      className={`tab-button ${inputMode === 'text' ? 'is-active' : ''}`}
                    >
                      Paste Text
                    </button>
                  </div>
                </div>

                {inputMode === 'file' ? (
                  <div className={`grid gap-4 ${isCompareMode ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
                    {/* File V1 */}
                    <div
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                      onClick={() => fileInputRef.current?.click()}
                      className={`dropzone cursor-pointer ${file ? 'has-file' : ''}`}
                    >
                      <input type="file" ref={fileInputRef} className="hidden" accept="image/*,application/pdf" onChange={handleFileChange} />
                      <div className="w-14 h-14 bg-white/80 rounded-full flex items-center justify-center text-teal-700 mx-auto mb-3">
                        <Upload className="w-7 h-7" />
                      </div>
                      <p className="text-base font-semibold text-slate-900 text-center">{file ? file.name : (isCompareMode ? "Upload Original" : "Click or drag to upload")}</p>
                      {!file && <p className="text-xs text-slate-500 mt-2 text-center">PDF, PNG, JPG accepted</p>}
                    </div>

                    {/* NEW: File V2 (Only shown in Compare Mode) */}
                    {isCompareMode && (
                      <div
                        onDragOver={handleDragOver}
                        onDrop={handleDropV2}
                        onClick={() => fileInputRefV2.current?.click()}
                        className={`dropzone cursor-pointer ${fileV2 ? 'has-file border-teal-500 bg-teal-50/50' : 'border-dashed'}`}
                      >
                        <input type="file" ref={fileInputRefV2} className="hidden" accept="image/*,application/pdf" onChange={handleFileChangeV2} />
                        <div className="w-14 h-14 bg-white/80 rounded-full flex items-center justify-center text-teal-700 mx-auto mb-3">
                          <Upload className="w-7 h-7" />
                        </div>
                        <p className="text-base font-semibold text-slate-900 text-center">{fileV2 ? fileV2.name : "Upload Revised"}</p>
                        {!fileV2 && <p className="text-xs text-slate-500 mt-2 text-center">Upload the newer version</p>}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className={`grid gap-4 ${isCompareMode ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
                    {/* Text V1 */}
                    <div className="card-muted rounded-2xl p-5">
                      <label className="block text-sm font-semibold text-slate-700 mb-2">
                        {isCompareMode ? "Original Contract Text" : "Paste Contract Text"}
                      </label>
                      <textarea
                        value={contractText} onChange={(e) => setContractText(e.target.value)} rows={8}
                        className="w-full resize-y rounded-lg border border-slate-200 px-4 py-3 text-sm text-slate-900 outline-none focus:ring-2 focus:ring-teal-600 focus:border-transparent"
                      />
                    </div>
                    
                    {/* NEW: Text V2 (Only shown in Compare Mode) */}
                    {isCompareMode && (
                      <div className="card-muted rounded-2xl p-5">
                        <label className="block text-sm font-semibold text-slate-700 mb-2">Revised Contract Text</label>
                        <textarea
                          value={contractTextV2} onChange={(e) => setContractTextV2(e.target.value)} rows={8}
                          className="w-full resize-y rounded-lg border border-slate-200 px-4 py-3 text-sm text-slate-900 outline-none focus:ring-2 focus:ring-teal-600 focus:border-transparent"
                        />
                      </div>
                    )}
                  </div>
                )}

                <div className="space-y-3">
                  <label className="block text-sm font-semibold text-slate-700">Contract Type</label>
                  <select value={contractType} onChange={(e) => setContractType(e.target.value)} className="w-full bg-white border border-slate-200 rounded-lg px-4 py-3 text-slate-900 outline-none focus:ring-2 focus:ring-teal-600">
                    {CONTRACT_TYPES.map(type => <option key={type.id} value={type.id}>{type.label}</option>)}
                  </select>
                </div>

                <div className="card-muted rounded-2xl p-5 space-y-4">
                  <h3 className="text-sm font-bold text-slate-800 border-b border-slate-200 pb-2">Your Guardrails</h3>
                  <div className="grid grid-cols-2 gap-4">
                    {activeFields.map(field => {
                      const value = dynamicReqs[field.key] !== undefined ? dynamicReqs[field.key] : (field.type === 'select' ? field.options?.[0] : (field.type === 'checkbox' ? false : ''));

                      if (field.type === 'text') {
                        return (
                          <div key={field.key}>
                            <label className="block text-xs font-semibold text-slate-600 mb-1">{field.label}</label>
                            <input type="text" value={value || ''} onChange={e => setDynamicReqs(prev => ({...prev, [field.key]: e.target.value}))} placeholder={field.placeholder} className="w-full border border-slate-200 rounded-md px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-teal-600"/>
                          </div>
                        );
                      } else if (field.type === 'select') {
                        return (
                          <div key={field.key}>
                            <label className="block text-xs font-semibold text-slate-600 mb-1">{field.label}</label>
                            <select value={value || ''} onChange={e => setDynamicReqs(prev => ({...prev, [field.key]: e.target.value}))} className="w-full border border-slate-200 rounded-md px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-teal-600">
                              {field.options?.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                            </select>
                          </div>
                        );
                      } else if (field.type === 'checkbox') {
                        return (
                          <div key={field.key} className="flex items-center justify-between px-2 pt-5">
                            <label className="text-xs font-semibold text-slate-600">{field.label}</label>
                            <input type="checkbox" checked={!!value} onChange={e => setDynamicReqs(prev => ({...prev, [field.key]: e.target.checked}))} className="w-4 h-4 rounded text-teal-600"/>
                          </div>
                        );
                      }
                      return null;
                    })}
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-600 mb-1">Additional Needs</label>
                    <textarea
                      value={requirements} onChange={(e) => setRequirements(e.target.value)} rows={2}
                      placeholder="Example: Need flexible hours for exams..."
                      className="w-full resize-y rounded-md border border-slate-200 px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-teal-600"
                    />
                  </div>
                </div>

                {!isCompareMode && (
                  <div className="card-muted rounded-2xl p-5 space-y-4">
                    <h3 className="text-sm font-bold text-slate-800 border-b border-slate-200 pb-2">Negotiation Details</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="relative">
                        <Building className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
                        <input type="text" value={companyName} onChange={e => setCompanyName(e.target.value)} placeholder="Company Name" className="w-full pl-9 pr-3 py-2 border border-slate-200 rounded-md text-sm outline-none focus:ring-1 focus:ring-teal-600"/>
                      </div>
                      <div className="relative">
                        <User className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
                        <input type="text" value={userName} onChange={e => setUserName(e.target.value)} placeholder="Your Full Name" className="w-full pl-9 pr-3 py-2 border border-slate-200 rounded-md text-sm outline-none focus:ring-1 focus:ring-teal-600"/>
                      </div>
                    </div>
                  </div>
                )}

                <div className="flex gap-2 flex-wrap items-center">
                  <span className="text-xs text-slate-500 font-medium">Try an example:</span>
                  {EXAMPLES.map(ex => (
                    <button
                      key={ex.label}
                      type="button"
                      onClick={() => {
                        setInputMode('text');
                        setContractText(ex.text);
                        setContractType(ex.type);
                        setIsCompareMode(false); // Disable compare to easily view example
                      }}
                      className="btn-ghost btn-ghost--small text-xs"
                    >
                      {ex.label}
                    </button>
                  ))}
                </div>

                <button
                  onClick={isCompareMode ? handleCompare : handleAnalyze}
                  disabled={isSubmitDisabled}
                  className="btn-primary w-full disabled:opacity-50"
                >
                  {isCompareMode
                    ? (isComparing ? 'Comparing Versions...' : 'Compare Contracts')
                    : (isAnalyzing ? 'Analyzing Pipeline...' : 'Analyze Contract')}
                </button>

                {/* Progress Indicators */}
                {(isAnalyzing || streamStatus?.stage === 'done') && !isCompareMode && (
                  <div className="w-full space-y-3 mt-2 animate-fade-in">
                    <div className="flex items-center gap-2">
                      {ANALYSIS_STEPS.map((step, index) => {
                        const isDone = activeStepIndex > index;
                        const isActive = activeStepIndex === index;
                        return (
                          <div key={step.key} className="flex items-center gap-2 flex-1">
                            <div
                              className={`step-dot transition ${
                                isDone ? 'bg-emerald-500' : isActive ? 'bg-teal-500 animate-pulse' : 'bg-slate-300'
                              }`}
                            />
                            <span className={`text-xs font-semibold uppercase tracking-wider ${isDone || isActive ? 'text-slate-700' : 'text-slate-400'}`}>
                              {step.label}
                            </span>
                            {index < ANALYSIS_STEPS.length - 1 && (
                              <div className={`h-px flex-1 ${isDone ? 'bg-emerald-400' : 'bg-slate-200'}`} />
                            )}
                          </div>
                        );
                      })}
                    </div>
                    <div className="text-xs text-slate-500 flex items-center gap-2">
                      <span className="inline-flex h-2 w-2 rounded-full bg-teal-500" />
                      {streamStatus?.message || 'Processing...'}
                    </div>
                  </div>
                )}
                
                {isComparing && (
                  <div className="text-xs text-teal-600 flex items-center justify-center gap-2 mt-4 font-semibold animate-pulse">
                    <Loader2 className="w-4 h-4 animate-spin" /> Concurrently analyzing revisions...
                  </div>
                )}

                {error && <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>}
              </div>
            </section>
          </div>

        ) : compareResult ? (
          
          /* --- NEW: COMPARE RESULT UI --- */
          <div className="space-y-8 export-root animate-fade-in">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-200 pb-4 gap-4 screen-only">
              <div>
                <h2 className="text-2xl font-bold flex items-center gap-3 text-slate-900">
                  Version Comparison
                  <span className={`text-xs px-2 py-1 rounded-full uppercase tracking-wider font-bold ${
                    compareResult.overall_change === 'IMPROVED' ? 'bg-green-100 text-green-700' :
                    compareResult.overall_change === 'WORSE' ? 'bg-red-100 text-red-700' :
                    'bg-slate-200 text-slate-700'
                  }`}>
                    {compareResult.overall_change}
                  </span>
                </h2>
                <p className="text-sm text-slate-500 mt-1">Comparing Original vs Revised flags</p>
              </div>
              <div>
                <button onClick={() => { setCompareResult(null); setIsCompareMode(false); setFile(null); setFileV2(null); setContractText(''); setContractTextV2(''); }} className="btn-secondary text-sm">
                  New Analysis
                </button>
              </div>
            </div>

            <div className="card summary-card">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Delta Summary</h3>
              </div>
              <p className="mt-3 text-sm text-slate-800 leading-relaxed font-medium">
                {compareResult.summary}
              </p>
            </div>

            <div className="grid gap-4 md:grid-cols-3 items-start">
              <div className="card border-l-4 border-l-green-500">
                <h4 className="text-sm font-bold text-green-800 mb-4 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4" /> Resolved Flags
                </h4>
                <ul className="space-y-2">
                  {compareResult.resolved_flags.length > 0 ? compareResult.resolved_flags.map((f, i) => (
                    <li key={i} className="text-sm text-slate-700 bg-green-50 px-3 py-2 rounded-md border border-green-100">{f}</li>
                  )) : <li className="text-sm text-slate-400 italic">No flags were resolved in this revision.</li>}
                </ul>
              </div>
              
              <div className="card border-l-4 border-l-red-500">
                <h4 className="text-sm font-bold text-red-800 mb-4 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" /> New Risks Introduced
                </h4>
                <ul className="space-y-2">
                  {compareResult.new_flags.length > 0 ? compareResult.new_flags.map((f, i) => (
                    <li key={i} className="text-sm text-slate-700 bg-red-50 px-3 py-2 rounded-md border border-red-100">{f}</li>
                  )) : <li className="text-sm text-slate-400 italic">No new risky clauses detected.</li>}
                </ul>
              </div>
              
              <div className="card border-l-4 border-l-amber-500">
                <h4 className="text-sm font-bold text-amber-800 mb-4 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" /> Remaining Flags
                </h4>
                <ul className="space-y-2">
                  {compareResult.remaining_flags.length > 0 ? compareResult.remaining_flags.map((f, i) => (
                    <li key={i} className="text-sm text-slate-700 bg-amber-50 px-3 py-2 rounded-md border border-amber-100">{f}</li>
                  )) : <li className="text-sm text-slate-400 italic">No flags remaining from the original.</li>}
                </ul>
              </div>
            </div>
          </div>

        ) : result ? (

          /* --- EXISTING: ANALYSIS RESULT UI --- */
          <div className="space-y-8 export-root animate-fade-in" ref={reportRef} data-export-root="true">
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
              {result.summary ? (
                <div className="pdf-note" style={{ marginTop: '10px' }}>
                  <strong>TL;DR:</strong> {result.summary}
                </div>
              ) : null}
            </div>

            <div className="flex items-center justify-between border-b border-slate-200 pb-4 screen-only" data-export-ignore="true">
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
                {result?.timing && (
                  <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-500">
                    <span className="rounded-full bg-slate-100 px-2 py-1">E4B: {formatMs(result.timing.e4b_ms)}</span>
                    <span className="rounded-full bg-slate-100 px-2 py-1">MoE: {formatMs(result.timing.moe_ms)}</span>
                    <span className="rounded-full bg-slate-100 px-2 py-1">Dense: {formatMs(result.timing.dense_ms)}</span>
                  </div>
                )}
              </div>
              <div className="flex gap-3">
                <button onClick={handleDownloadPDF} disabled={isAnalyzing || isExporting} className="btn-secondary flex items-center gap-2 disabled:opacity-50">
                  {isExporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                  {isExporting ? 'Exporting…' : 'Export PDF'}
                </button>
                <button onClick={() => { setResult(null); setFile(null); setError(null); setContractText(''); }} className="btn-ghost text-sm">
                  New Analysis
                </button>
              </div>
            </div>

            {error && (
              <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            )}

            {shareUrl ? (
              <div className="card screen-only">
                <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">Shareable Report Link</div>
                <div className="mt-2 flex flex-col gap-2 sm:flex-row sm:items-center">
                  <input
                    type="text"
                    readOnly
                    value={shareUrl}
                    className="flex-1 rounded-md border border-slate-200 px-3 py-2 text-xs text-slate-700"
                  />
                  <button
                    type="button"
                    onClick={() => navigator.clipboard.writeText(shareUrl)}
                    className="btn-secondary text-xs"
                  >
                    Copy Link
                  </button>
                </div>
              </div>
            ) : null}

            <div className="report-hero card screen-only reveal">
              <div className="report-hero-main">
                <div className="section-title">Executive snapshot</div>
                <div className="report-verdict">
                  <span className={`verdict-chip ${verdictToneClass}`}>{verdictMeta?.label ?? 'PENDING'}</span>
                  <span className="report-verdict-text">{verdictMeta?.description ?? 'Verdict pending.'}</span>
                </div>
                <p className="report-reason">
                  {result.verdict_reason || (isAnalyzing ? 'Computing verdict...' : 'Verdict pending.')}
                </p>
                <div className="report-meta">
                  <span className="pill">Contract: {CONTRACT_TYPES.find(c => c.id === contractType)?.label || 'General Contract'}</span>
                  <span className="pill">Share-ready report</span>
                </div>
              </div>
              <div className="report-hero-scores">
                <div className="score-block">
                  <div className="score-label">Risk score</div>
                  <div className="score-row">
                    <span className="score-value">{result.risk_score !== null ? result.risk_score : '--'}</span>
                    <span className="score-unit">/ 100</span>
                  </div>
                  <div className="score-bar">
                    <span style={{ width: `${riskPercent}%` }} />
                  </div>
                  <div className="score-foot">{riskMeta.description}</div>
                  {benchmarkNote && <div className="score-note">{benchmarkNote}</div>}
                </div>
                <div className="score-block">
                  <div className="score-label">Compatibility</div>
                  <div className="score-row">
                    <span className="score-value">{result.compatibility_score !== null ? result.compatibility_score : '--'}</span>
                    <span className="score-unit">/ 100</span>
                  </div>
                  <div className="score-bar">
                    <span style={{ width: `${compatibilityPercent}%` }} />
                  </div>
                  <div className="score-foot">{compatibilityMeta.description}</div>
                </div>
              </div>
            </div>

            <div className="card summary-card">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">TL;DR Summary</h3>
                <span className="text-[11px] font-semibold text-slate-400">3-sentence brief</span>
              </div>
              <p className="mt-3 text-sm text-slate-700 leading-relaxed serif">
                {result.summary || (isAnalyzing ? 'Generating summary...' : 'Summary not available.')}
              </p>
            </div>

            <div className="report-grid screen-only reveal">
              {/* Risk Score */}
              <div className="card flex flex-col justify-between">
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
                  {benchmarkNote && (
                    <p className="mt-1 text-[11px] text-slate-400 font-medium">
                      {benchmarkNote}
                    </p>
                  )}
                </div>
              </div>

              {/* Compatibility Score & Checklist */}
              <div className="card flex flex-col">
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
              <div className="card flex flex-col">
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
                    <div key={i} className="clause-card transition-all hover:border-red-300">
                      <div 
                        className="clause-header flex justify-between items-start cursor-pointer"
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

                          {flag.suggested_rewrite && (
                            <div className="bg-emerald-50 border border-emerald-200 p-4 rounded-lg flex items-start justify-between gap-3">
                              <div className="text-emerald-800 text-sm">
                                <span className="font-bold uppercase text-xs tracking-wider block mb-1">Suggested rewrite</span>
                                {flag.suggested_rewrite}
                              </div>
                              <button
                                type="button"
                                onClick={() => navigator.clipboard.writeText(flag.suggested_rewrite || '')}
                                className="text-xs font-semibold text-emerald-700 hover:text-emerald-900 flex items-center gap-1"
                                data-export-ignore="true"
                              >
                                <Copy className="w-4 h-4" /> Copy
                              </button>
                            </div>
                          )}

                          <div className="border border-slate-200 rounded-lg p-4 bg-slate-50">
                            <div className="flex items-center justify-between mb-3">
                              <h5 className="text-xs font-bold uppercase tracking-wider text-slate-600">Ask about this clause</h5>
                              <span className="text-[10px] text-slate-400">Gemma Q&A</span>
                            </div>
                            <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                              {(clauseChats[i]?.messages || []).length === 0 ? (
                                <p className="text-xs text-slate-500">Ask a follow-up question about this clause.</p>
                              ) : (
                                (clauseChats[i]?.messages || []).map((msg, idx) => (
                                  <div
                                    key={`chat-${i}-${idx}`}
                                    className={`rounded-lg px-3 py-2 text-xs leading-relaxed ${
                                      msg.role === 'user'
                                        ? 'bg-white border border-slate-200 text-slate-700'
                                        : 'bg-blue-50 border border-blue-100 text-slate-700'
                                    }`}
                                  >
                                    <span className="font-semibold mr-2">
                                      {msg.role === 'user' ? 'You' : 'Gemma'}:
                                    </span>
                                    {msg.content}
                                  </div>
                                ))
                              )}
                            </div>
                            {clauseChats[i]?.error ? (
                              <div className="mt-2 text-xs text-red-600">{clauseChats[i]?.error}</div>
                            ) : null}
                            <div className="mt-3 flex items-center gap-2">
                              <input
                                type="text"
                                value={clauseChats[i]?.input || ''}
                                onChange={(e) => updateClauseChat(i, { input: e.target.value })}
                                placeholder="Ask a follow-up question..."
                                className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-xs outline-none focus:ring-1 focus:ring-blue-500"
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') {
                                    e.preventDefault();
                                    handleAskClause(i);
                                  }
                                }}
                              />
                              <button
                                type="button"
                                onClick={() => handleAskClause(i)}
                                disabled={clauseChats[i]?.isLoading}
                                className="btn-secondary text-xs disabled:opacity-50"
                              >
                                {clauseChats[i]?.isLoading ? 'Asking…' : 'Ask'}
                              </button>
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
                <div className="space-y-4 mt-8 card negotiation-block">
                  <div className="flex items-center justify-between border-b border-blue-200 pb-2 negotiation-header">
                    <h3 className="font-bold flex items-center gap-2 text-blue-900">
                      Draft Negotiation Email
                      <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">Dense model</span>
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
                   <div className="bg-white border border-blue-200 rounded-lg p-5 text-sm text-slate-700 whitespace-pre-wrap serif leading-relaxed shadow-inner negotiation-body">
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
            
            <p className="text-center text-xs text-slate-400 pt-4 pb-2">
              FinePrint uses AI and does not constitute legal advice. 
              Consult a qualified lawyer before signing any binding agreement.
            </p>

          </div>
        ) : null}
      </main>
    </div>
  );
}

export default App;