import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, 
  Settings2, 
  TrendingUp, 
  HelpCircle, 
  Send, 
  Terminal, 
  Bot, 
  User, 
  CheckCircle, 
  AlertTriangle, 
  Sliders, 
  Upload, 
  Database, 
  Cpu, 
  ChevronRight, 
  RefreshCw, 
  Download,
  BookOpen,
  Info
} from 'lucide-react';
import { datasets } from './data/datasets';
import { simulateModelTraining, estimateTelecomChurnRisk, ClientSideMLEngine } from './utils/mlEngine';
import { ModelType, Hyperparameters, TrainingMetrics, CustomerInput, Message } from './types';

export default function App() {
  // Navigation Tabs
  const [activeTab, setActiveTab] = useState<'simulator' | 'csv-trainer' | 'single-pred' | 'ai-expert'>('simulator');

  // --- TAB 1: MODEL SIMULATOR STATE ---
  const [selectedDataset, setSelectedDataset] = useState(datasets[0]);
  const [selectedModel, setSelectedModel] = useState<ModelType>('gradient_boosting');
  const [hyperparams, setHyperparams] = useState<Hyperparameters>({
    learningRate: 0.1,
    numEstimators: 200,
    maxDepth: 6,
    regularization: 'l2'
  });
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [isTraining, setIsTraining] = useState(false);

  // --- TAB 2: CSV CLIENT-SIDE TRAINER STATE ---
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [csvPreview, setCsvPreview] = useState<{ headers: string[]; rows: any[] } | null>(null);
  const [csvProfile, setCsvProfile] = useState<any>(null);
  const [targetColumn, setTargetColumn] = useState<string>('');
  const [csvLr, setCsvLr] = useState<number>(0.1);
  const [csvEpochs, setCsvEpochs] = useState<number>(50);
  const [csvMetrics, setCsvMetrics] = useState<TrainingMetrics | null>(null);
  const [csvTrainingError, setCsvTrainingError] = useState<string | null>(null);
  const [isCsvTraining, setIsCsvTraining] = useState(false);

  // --- TAB 3: SINGLE CUSTOMER PREDICTION STATE ---
  const [customerInput, setCustomerInput] = useState<CustomerInput>({
    tenure: 12,
    contract: 'month-to-month',
    monthlyCharges: 75,
    techSupport: 'no',
    paymentMethod: 'electronic_check',
    paperlessBilling: 'yes',
    internetService: 'fiber_optic',
    multipleLines: 'no',
    partner: 'no',
    dependents: 'no',
    seniorCitizen: 'no'
  });
  const [predictionOutcome, setPredictionOutcome] = useState<any>(null);

  // --- TAB 4: CHAT WITH AI CONSULTANT STATE ---
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: "Hello! I am your AI Machine Learning Consultant. I specialize in customer churn prediction. I can explain ML algorithms (XGBoost, Random Forests, SMOTE, Categorical Target Encoding), debug code, or write production-ready Python templates. Try asking me:\n\n* 'Write a complete Python script to load telecom churn data, handle class imbalance, and train an XGBoost model.'\n* 'Explain how to handle missing values and scale numeric features in a pipeline.'",
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isAiLoading, setIsAiLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Simulate default model training on load
  useEffect(() => {
    handleSimulateTraining();
  }, [selectedDataset, selectedModel]);

  // Recalculate single prediction on change
  useEffect(() => {
    const outcome = estimateTelecomChurnRisk(customerInput);
    setPredictionOutcome(outcome);
  }, [customerInput]);

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSimulateTraining = () => {
    setIsTraining(true);
    setTimeout(() => {
      const res = simulateModelTraining(selectedModel, hyperparams, selectedDataset.id);
      setMetrics(res);
      setIsTraining(false);
    }, 800);
  };

  // CSV Drag and drop handling
  const handleCsvUpload = (text: string, file: File) => {
    try {
      const { headers, rows } = ClientSideMLEngine.parseCSV(text);
      if (headers.length === 0 || rows.length === 0) {
        alert("The uploaded file seems empty or is not a valid CSV.");
        return;
      }
      setCsvFile(file);
      setCsvPreview({ headers, rows });
      
      const target = ClientSideMLEngine.findTargetColumn(headers) || headers[headers.length - 1] || '';
      setTargetColumn(target);
      
      const profile = ClientSideMLEngine.profileDataset(headers, rows, target);
      setCsvProfile(profile);
      setCsvMetrics(null);
      setCsvTrainingError(null);
    } catch (e: any) {
      alert("Error parsing CSV: " + e.message);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          handleCsvUpload(event.target.result as string, file);
        }
      };
      reader.readAsText(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          handleCsvUpload(event.target.result as string, file);
        }
      };
      reader.readAsText(file);
    }
  };

  const handleClientSideTrain = () => {
    if (!csvPreview || !targetColumn) return;
    setIsCsvTraining(true);
    setCsvTrainingError(null);
    setTimeout(() => {
      const result = ClientSideMLEngine.trainLogisticRegression(
        csvPreview.rows,
        targetColumn,
        csvPreview.headers,
        csvLr,
        csvEpochs
      );
      if (result.success) {
        setCsvMetrics(result.metrics);
      } else {
        setCsvTrainingError(result.error || 'Training failed.');
      }
      setIsCsvTraining(false);
    }, 1200);
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputText,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMsg]);
    setInputText('');
    setIsAiLoading(true);

    // Context helper
    const contextPrompt = `
      User wants to build churn prediction using Python.
      Currently simulated dataset: ${selectedDataset.name} (${selectedDataset.industry}).
      Simulated model: ${selectedModel} (AUC: ${metrics?.rocAuc || 'N/A'}, Accuracy: ${metrics?.accuracy || 'N/A'}).
      User prompt: ${userMsg.content}
    `;

    try {
      const response = await fetch('/api/assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, userMsg].map(m => ({ role: m.role, content: m.content }))
        })
      });

      if (!response.ok) {
        throw new Error('Server returned error');
      }

      const data = await response.json();
      const botMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.content || "I couldn't process that command. Let me know if you want to try again.",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: "I ran into an issue connecting to the Gemini client. Please ensure you have configured your `GEMINI_API_KEY` in **Settings > Secrets** in AI Studio.",
          timestamp: new Date()
        }
      ]);
    } finally {
      setIsAiLoading(false);
    }
  };

  // Helper to trigger canned prompt in AI tab
  const askAIPythonTemplate = (codeType: string) => {
    setActiveTab('ai-expert');
    let prompt = '';
    if (codeType === 'pipeline') {
      prompt = `Write a complete, professional Python script using scikit-learn and pandas to train a Customer Churn Model. Implement:
1. Handling missing values with simple imputer
2. One-hot encoding of categorical variables (like Contract, TechSupport)
3. StandardScaler for numeric features (like tenure, MonthlyCharges)
4. Train/test split and a LogisticRegression or RandomForest classifier
5. Calculate precision, recall, confusion matrix, and ROC-AUC metrics.`;
    } else if (codeType === 'imbalance') {
      prompt = `Show me how to address severe class imbalance in Customer Churn datasets using Python.
Provide code demonstrating SMOTE (Synthetic Minority Over-sampling Technique) from the imbalanced-learn package, and explain why adjusting decision thresholds is often better than oversampling.`;
    } else if (codeType === 'boosting') {
      prompt = `Provide a premium python code block using XGBoost and Optuna for hyperparameter tuning. Include early stopping on validation loss to prevent overfitting.`;
    }
    setInputText(prompt);
  };

  return (
    <div id="churn-workspace" className="min-h-screen bg-[#0A0A0A] text-[#E0E0E0] font-sans flex flex-col selection:bg-[#C5B48C] selection:text-[#0A0A0A]">
      
      {/* HEADER SECTION */}
      <header id="header-brand" className="h-20 border-b border-[#2A2A2A] px-6 lg:px-10 flex items-center justify-between bg-[#080808]">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 bg-gradient-to-tr from-[#8E7D5B] to-[#C5B48C] rounded-sm rotate-45 flex items-center justify-center shadow-lg shadow-[#C5B48C]/10">
            <span className="text-[10px] font-bold text-black -rotate-45 font-mono">ML</span>
          </div>
          <div>
            <h1 className="text-xl font-serif italic tracking-wide text-[#C5B48C]">Vantage Analytics</h1>
            <p className="text-[9px] text-[#555555] uppercase tracking-widest font-mono">Customer Churn Intelligence Engine</p>
          </div>
        </div>
        
        {/* Navigation Tabs */}
        <nav className="flex gap-4 sm:gap-6 text-xs uppercase tracking-[0.15em] font-medium text-[#888888]">
          <button 
            id="tab-simulator"
            onClick={() => setActiveTab('simulator')}
            className={`pb-2 transition-all ${activeTab === 'simulator' ? 'text-[#C5B48C] border-b-2 border-[#C5B48C]' : 'hover:text-white'}`}
          >
            Model Simulator
          </button>
          <button 
            id="tab-csv"
            onClick={() => setActiveTab('csv-trainer')}
            className={`pb-2 transition-all ${activeTab === 'csv-trainer' ? 'text-[#C5B48C] border-b-2 border-[#C5B48C]' : 'hover:text-white'}`}
          >
            CSV Live Trainer
          </button>
          <button 
            id="tab-single"
            onClick={() => setActiveTab('single-pred')}
            className={`pb-2 transition-all ${activeTab === 'single-pred' ? 'text-[#C5B48C] border-b-2 border-[#C5B48C]' : 'hover:text-white'}`}
          >
            Single Prediction
          </button>
          <button 
            id="tab-ai"
            onClick={() => setActiveTab('ai-expert')}
            className={`pb-2 transition-all ${activeTab === 'ai-expert' ? 'text-[#C5B48C] border-b-2 border-[#C5B48C]' : 'hover:text-white'}`}
          >
            AI Python Expert
          </button>
        </nav>
      </header>

      {/* SUB-HEADER OR STATUS RAIL */}
      <div className="bg-[#0D0D0D] border-b border-[#2A2A2A] px-6 lg:px-10 py-2.5 flex flex-wrap items-center justify-between text-xs text-[#888888]">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
            <span className="font-mono text-[11px] text-[#C5B48C]">Python 3.11 Workspace Active</span>
          </div>
          <div className="hidden md:flex items-center gap-1.5 border-l border-[#222222] pl-6">
            <span className="text-stone-600">Model Framework:</span>
            <span className="font-mono text-stone-300">scikit-learn, xgboost, lightgbm</span>
          </div>
        </div>
        <div className="flex items-center gap-4 text-[11px]">
          <span>Refreshed: <span className="font-mono text-stone-300">2026-06-25 // UTC</span></span>
          <span className="text-[#C5B48C]/70">Secure Sandbox Enclave</span>
        </div>
      </div>

      {/* MAIN VIEWPORT COMPONENT */}
      <main className="flex-1 grid grid-cols-1 md:grid-cols-12 gap-px bg-[#2A2A2A]">
        
        {/* ==================================== TAB 1: MODEL SIMULATOR ==================================== */}
        {activeTab === 'simulator' && (
          <>
            {/* Sidebar Controls - Left Column */}
            <aside id="simulator-sidebar" className="md:col-span-3 bg-[#0D0D0D] p-6 lg:p-8 flex flex-col gap-8 border-r border-[#2A2A2A]">
              
              {/* Dataset Selection */}
              <div>
                <h3 className="text-[10px] uppercase tracking-[0.2em] text-[#555555] mb-4 font-bold flex items-center gap-2">
                  <Database className="w-3.5 h-3.5 text-[#C5B48C]" /> 1. Select Dataset
                </h3>
                <div className="space-y-2">
                  {datasets.map(d => (
                    <button
                      key={d.id}
                      onClick={() => setSelectedDataset(d)}
                      className={`w-full text-left p-3 rounded-sm border transition-all ${
                        selectedDataset.id === d.id 
                          ? 'border-[#C5B48C] bg-[#C5B48C]/5 text-white' 
                          : 'border-[#2A2A2A] bg-[#121212]/50 text-stone-400 hover:text-white hover:border-[#444444]'
                      }`}
                    >
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs font-semibold">{d.name}</span>
                        <span className="text-[9px] bg-[#1A1A1A] px-1.5 py-0.5 rounded text-stone-400 border border-[#222]">
                          {d.industry}
                        </span>
                      </div>
                      <p className="text-[10px] line-clamp-2 text-stone-500 leading-relaxed">{d.description}</p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Algorithm selector */}
              <div>
                <h3 className="text-[10px] uppercase tracking-[0.2em] text-[#555555] mb-4 font-bold flex items-center gap-2">
                  <Cpu className="w-3.5 h-3.5 text-[#C5B48C]" /> 2. Model Algorithm
                </h3>
                <div className="space-y-1">
                  {[
                    { id: 'gradient_boosting', label: 'Gradient Boosting (XGBoost)', desc: 'High accuracy, handles non-linear patterns.' },
                    { id: 'random_forest', label: 'Random Forest Ensemble', desc: 'Robust ensemble, reduces variance, less overfitting.' },
                    { id: 'logistic_regression', label: 'Logistic Regression', desc: 'Fast, linear coefficients, highly interpretable.' },
                    { id: 'decision_tree', label: 'Decision Tree Classifier', desc: 'Hierarchical rules, visualizable splits.' }
                  ].map(m => (
                    <button
                      key={m.id}
                      onClick={() => setSelectedModel(m.id as ModelType)}
                      className={`w-full text-left p-2.5 rounded-sm border transition-all ${
                        selectedModel === m.id 
                          ? 'border-[#C5B48C] bg-[#1A1813] text-white' 
                          : 'border-transparent text-stone-400 hover:text-white hover:bg-[#121212]'
                      }`}
                    >
                      <div className="text-xs font-medium">{m.label}</div>
                      <div className="text-[9px] text-stone-500">{m.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Hyperparameters Form */}
              <div>
                <h3 className="text-[10px] uppercase tracking-[0.2em] text-[#555555] mb-4 font-bold flex items-center gap-2">
                  <Sliders className="w-3.5 h-3.5 text-[#C5B48C]" /> 3. Hyperparameters
                </h3>
                <div className="space-y-4 bg-[#121212] p-4 rounded border border-[#222] text-xs">
                  
                  {/* Learning Rate - if applicable */}
                  {selectedModel !== 'decision_tree' && (
                    <div>
                      <div className="flex justify-between text-[11px] mb-1">
                        <span className="text-stone-400">Learning Rate (Alpha)</span>
                        <span className="font-mono text-[#C5B48C]">{hyperparams.learningRate}</span>
                      </div>
                      <input 
                        type="range" 
                        min="0.01" 
                        max="0.5" 
                        step="0.01" 
                        value={hyperparams.learningRate || 0.1}
                        onChange={(e) => setHyperparams(prev => ({ ...prev, learningRate: parseFloat(e.target.value) }))}
                        className="w-full accent-[#C5B48C] bg-[#222] h-1 rounded"
                      />
                      <div className="flex justify-between text-[8px] text-stone-600 font-mono mt-1">
                        <span>0.01 (Slow)</span>
                        <span>0.5 (Aggressive)</span>
                      </div>
                    </div>
                  )}

                  {/* Num Estimators */}
                  {(selectedModel === 'random_forest' || selectedModel === 'gradient_boosting') && (
                    <div>
                      <div className="flex justify-between text-[11px] mb-1">
                        <span className="text-stone-400">Number of Trees</span>
                        <span className="font-mono text-[#C5B48C]">{hyperparams.numEstimators}</span>
                      </div>
                      <input 
                        type="range" 
                        min="20" 
                        max="500" 
                        step="10" 
                        value={hyperparams.numEstimators || 100}
                        onChange={(e) => setHyperparams(prev => ({ ...prev, numEstimators: parseInt(e.target.value) }))}
                        className="w-full accent-[#C5B48C] bg-[#222] h-1 rounded"
                      />
                      <div className="flex justify-between text-[8px] text-stone-600 font-mono mt-1">
                        <span>20 trees</span>
                        <span>500 trees</span>
                      </div>
                    </div>
                  )}

                  {/* Max Depth */}
                  {selectedModel !== 'logistic_regression' && (
                    <div>
                      <div className="flex justify-between text-[11px] mb-1">
                        <span className="text-stone-400">Max Depth (Tree Complexity)</span>
                        <span className="font-mono text-[#C5B48C]">{hyperparams.maxDepth}</span>
                      </div>
                      <input 
                        type="range" 
                        min="2" 
                        max="16" 
                        step="1" 
                        value={hyperparams.maxDepth || 6}
                        onChange={(e) => setHyperparams(prev => ({ ...prev, maxDepth: parseInt(e.target.value) }))}
                        className="w-full accent-[#C5B48C] bg-[#222] h-1 rounded"
                      />
                      <div className="flex justify-between text-[8px] text-stone-600 font-mono mt-1">
                        <span>2 (Stump)</span>
                        <span>16 (Complex / Overfit)</span>
                      </div>
                    </div>
                  )}

                  {/* Regularization Penalty */}
                  {selectedModel === 'logistic_regression' && (
                    <div>
                      <span className="text-stone-400 block mb-1.5">Regularization Penalty</span>
                      <div className="grid grid-cols-3 gap-1">
                        {['none', 'l1', 'l2'].map(p => (
                          <button
                            key={p}
                            onClick={() => setHyperparams(prev => ({ ...prev, regularization: p as any }))}
                            className={`p-1.5 text-[10px] uppercase font-mono rounded text-center border ${
                              hyperparams.regularization === p
                                ? 'border-[#C5B48C] bg-[#C5B48C]/10 text-[#C5B48C]'
                                : 'border-[#2A2A2A] bg-black text-stone-500 hover:text-stone-300'
                            }`}
                          >
                            {p}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  <button
                    onClick={handleSimulateTraining}
                    disabled={isTraining}
                    className="w-full mt-2 py-2.5 bg-gradient-to-r from-[#8E7D5B] to-[#C5B48C] text-black font-semibold rounded-sm text-xs uppercase tracking-wider flex items-center justify-center gap-2 hover:opacity-90 active:scale-95 transition-all cursor-pointer"
                  >
                    {isTraining ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        Fitting Estimator...
                      </>
                    ) : (
                      <>
                        <Play className="w-3.5 h-3.5 fill-black" />
                        Train & Evaluate
                      </>
                    )}
                  </button>

                </div>
              </div>

              {/* Helpful tips panel */}
              <div className="mt-auto p-4 border border-[#2A2A2A] bg-[#0A0A0A] rounded text-stone-400">
                <div className="flex items-start gap-2.5">
                  <Info className="w-4 h-4 text-[#C5B48C] shrink-0 mt-0.5" />
                  <div className="text-[10px] leading-relaxed">
                    <p className="font-serif italic text-[#C5B48C] mb-1">Expert Intuition:</p>
                    {selectedModel === 'decision_tree' && "Decision trees are prone to high variance. Increase regularization or choose Random Forest to reduce test metrics decay."}
                    {selectedModel === 'gradient_boosting' && "XGBoost builds trees sequentially. Too high a depth with high estimator counts can cause validation loss divergence."}
                    {selectedModel === 'logistic_regression' && "L1 regularization acts as sparse feature selection, driving non-essential coefficients to absolute zero."}
                    {selectedModel === 'random_forest' && "Ensembles are incredibly robust. Increasing estimators won't overfit, but adds standard train-time latency."}
                  </div>
                </div>
              </div>

            </aside>

            {/* Central Intelligence / Output Dashboard - Right Column */}
            <div id="simulator-main" className="md:col-span-9 bg-[#0A0A0A] p-6 lg:p-10 flex flex-col overflow-y-auto">
              
              {/* Header block inside */}
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4 mb-8 border-b border-[#1A1A1A] pb-6">
                <div>
                  <p className="text-[10px] uppercase tracking-[0.3em] text-[#555555] mb-2">Simulated Machine Learning Environment</p>
                  <h2 className="text-3xl lg:text-4xl font-serif italic text-white leading-tight">
                    Model: <span className="text-[#C5B48C] capitalize">{selectedModel.replace('_', ' ')}</span>
                  </h2>
                  <p className="text-xs text-stone-400 max-w-xl mt-1">
                    Evaluating <span className="text-stone-200">{selectedDataset.name}</span> with a 80-20 train-test cross-validation split. Adjust parameters on the left to inspect performance tradeoffs.
                  </p>
                </div>
                
                {/* Global KPI banner */}
                <div className="text-left sm:text-right shrink-0 bg-[#121212] px-6 py-3 rounded border border-[#222]">
                  <p className="text-[9px] uppercase tracking-[0.2em] text-[#888888] mb-1">ROC-AUC SCORE</p>
                  <span className="text-3xl font-light tracking-tighter text-white">
                    {metrics ? metrics.rocAuc.toFixed(3) : '---'}
                    <span className="text-sm text-[#C5B48C] ml-1.5">AUC</span>
                  </span>
                </div>
              </div>

              {/* METRIC CARD BAR GRID */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                {[
                  { name: 'Model Accuracy', value: metrics?.accuracy, desc: 'Ratio of true predictions' },
                  { name: 'Precision (PPV)', value: metrics?.precision, desc: 'Avoid false positive alerts' },
                  { name: 'Recall (Sensitivity)', value: metrics?.recall, desc: 'Capture true churned users' },
                  { name: 'F1-Harmonic Mean', value: metrics?.f1Score, desc: 'Balanced measure of P & R' }
                ].map((m, idx) => (
                  <div key={idx} className="p-4 border border-[#2A2A2A] bg-gradient-to-r from-[#121212] to-transparent rounded-sm">
                    <p className="text-[9px] uppercase tracking-widest text-[#888888] mb-1.5">{m.name}</p>
                    <p className="text-2xl font-light tracking-tight text-white">
                      {m.value !== undefined ? m.value.toFixed(3) : '0.000'}
                    </p>
                    <p className="text-[9px] text-[#555555] mt-1">{m.desc}</p>
                    <div className="w-full bg-[#1A1A1A] h-0.5 mt-3">
                      <div 
                        className="bg-[#C5B48C] h-full transition-all duration-1000" 
                        style={{ width: `${(m.value || 0) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>

              {/* DATA VISUALIZATION GRID: ROC, FEAT IMPORTANCE, LOSS CURVE */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-8">
                
                {/* Left Graph: ROC Curve */}
                <div className="lg:col-span-4 p-5 border border-[#2A2A2A] bg-[#0E0E0E] rounded flex flex-col">
                  <div className="flex justify-between items-center mb-4">
                    <h4 className="text-[10px] uppercase tracking-[0.2em] text-[#888888] font-bold">Receiver Operating Characteristic (ROC)</h4>
                    <HelpCircle className="w-3.5 h-3.5 text-stone-600 hover:text-stone-400 cursor-pointer" title="True Positive Rate vs. False Positive Rate" />
                  </div>
                  
                  {/* SVG ROC Plot */}
                  <div className="flex-1 min-h-[200px] bg-black/60 relative p-2 border border-[#1A1A1A] flex items-center justify-center">
                    {metrics && (
                      <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible">
                        {/* Grid lines */}
                        <line x1="0" y1="0" x2="100" y2="100" stroke="#222" strokeDasharray="2" />
                        <line x1="0" y1="100" x2="100" y2="100" stroke="#333" strokeWidth="0.5" />
                        <line x1="0" y1="0" x2="0" y2="100" stroke="#333" strokeWidth="0.5" />
                        
                        {/* ROC Curve Polyline */}
                        <polyline
                          fill="none"
                          stroke="#C5B48C"
                          strokeWidth="2"
                          points={metrics.rocCurve.map(pt => `${pt.fpr * 100},${100 - pt.tpr * 100}`).join(' ')}
                        />
                        {/* Dots */}
                        {metrics.rocCurve.map((pt, i) => (
                          <circle
                            key={i}
                            cx={pt.fpr * 100}
                            cy={100 - pt.tpr * 100}
                            r="1.5"
                            fill="#FFF"
                            className="hover:r-3 transition-all cursor-crosshair"
                          >
                            <title>FPR: {pt.fpr}, TPR: {pt.tpr}</title>
                          </circle>
                        ))}
                      </svg>
                    )}
                    <div className="absolute left-1 top-1 text-[8px] text-[#555] font-mono">TPR (Recall)</div>
                    <div className="absolute right-1 bottom-1 text-[8px] text-[#555] font-mono">FPR</div>
                  </div>
                  <p className="text-[9px] text-stone-500 mt-2 text-center">
                    Higher curve bulge (closer to top-left) indicates greater discriminative power between Churn vs. Retained.
                  </p>
                </div>

                {/* Center Graph: Train vs Val Loss Curve */}
                <div className="lg:col-span-4 p-5 border border-[#2A2A2A] bg-[#0E0E0E] rounded flex flex-col">
                  <h4 className="text-[10px] uppercase tracking-[0.2em] text-[#888888] mb-4 font-bold">Optimization Loss Curve</h4>
                  
                  {/* SVG Line Graph */}
                  <div className="flex-1 min-h-[200px] bg-black/60 relative p-2 border border-[#1A1A1A] flex items-center justify-center">
                    {metrics && (
                      <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible" preserveAspectRatio="none">
                        {/* Grid lines */}
                        <line x1="0" y1="50" x2="100" y2="50" stroke="#1C1C1C" strokeWidth="0.5" />
                        <line x1="0" y1="100" x2="100" y2="100" stroke="#333" strokeWidth="0.5" />
                        <line x1="0" y1="0" x2="0" y2="100" stroke="#333" strokeWidth="0.5" />
                        
                        {/* Training Loss Line */}
                        <polyline
                          fill="none"
                          stroke="#888888"
                          strokeWidth="1"
                          strokeDasharray="1 1"
                          points={metrics.lossCurve.map((lc, idx) => {
                            const x = (idx / (metrics.lossCurve.length - 1)) * 100;
                            const y = 100 - (lc.loss / 0.9) * 100;
                            return `${x},${y}`;
                          }).join(' ')}
                        />

                        {/* Validation Loss Line */}
                        <polyline
                          fill="none"
                          stroke="#C5B48C"
                          strokeWidth="1.5"
                          points={metrics.lossCurve.map((lc, idx) => {
                            const x = (idx / (metrics.lossCurve.length - 1)) * 100;
                            const y = 100 - (lc.valLoss / 0.9) * 100;
                            return `${x},${y}`;
                          }).join(' ')}
                        />
                      </svg>
                    )}
                    <div className="absolute right-2 top-2 flex flex-col gap-1 text-[8px] font-mono text-stone-500">
                      <div className="flex items-center gap-1">
                        <span className="w-2.5 h-0.5 bg-stone-500 inline-block"></span> Train Loss
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="w-2.5 h-0.5 bg-[#C5B48C] inline-block"></span> Val Loss
                      </div>
                    </div>
                    <div className="absolute left-1 top-1 text-[8px] text-[#555] font-mono">Loss</div>
                    <div className="absolute right-1 bottom-1 text-[8px] text-[#555] font-mono">Epochs</div>
                  </div>
                  <p className="text-[9px] text-stone-500 mt-2 text-center">
                    Validation divergence (when validation loss goes back up while training loss decreases) simulates overfitting.
                  </p>
                </div>

                {/* Right Panel: Confusion Matrix & Data Segments */}
                <div className="lg:col-span-4 p-5 border border-[#2A2A2A] bg-[#0E0E0E] rounded flex flex-col justify-between">
                  <div>
                    <h4 className="text-[10px] uppercase tracking-[0.2em] text-[#888888] mb-4 font-bold">Standard Confusion Matrix (N=1000)</h4>
                    {metrics && (
                      <div className="grid grid-cols-2 gap-2 text-center text-xs font-mono">
                        
                        <div className="p-3 bg-[#121212] border border-[#222] rounded">
                          <span className="text-[9px] text-[#555] uppercase block mb-1">True Negative (TN)</span>
                          <span className="text-lg text-stone-300 font-light">{metrics.confusionMatrix.trueNegative}</span>
                          <span className="text-[8px] block text-green-700 font-bold uppercase mt-1">Retained Correctly</span>
                        </div>

                        <div className="p-3 bg-[#1D1212] border border-red-950/40 rounded">
                          <span className="text-[9px] text-red-950 block mb-1">False Positive (FP)</span>
                          <span className="text-lg text-red-400 font-light">{metrics.confusionMatrix.falsePositive}</span>
                          <span className="text-[8px] block text-red-500/80 uppercase mt-1">False Alarm Alert</span>
                        </div>

                        <div className="p-3 bg-[#1D1212] border border-red-950/40 rounded">
                          <span className="text-[9px] text-red-950 block mb-1">False Negative (FN)</span>
                          <span className="text-lg text-red-400 font-light">{metrics.confusionMatrix.falseNegative}</span>
                          <span className="text-[8px] block text-red-500/80 uppercase mt-1">Missed Churn Opportunity</span>
                        </div>

                        <div className="p-3 bg-[#1A1813] border border-[#C5B48C]/20 rounded">
                          <span className="text-[9px] text-[#C5B48C]/80 block mb-1">True Positive (TP)</span>
                          <span className="text-lg text-[#C5B48C] font-semibold">{metrics.confusionMatrix.truePositive}</span>
                          <span className="text-[8px] block text-[#C5B48C] font-bold uppercase mt-1">Churn Correctly Caught</span>
                        </div>

                      </div>
                    )}
                  </div>
                  
                  <div className="mt-4 border-t border-[#1A1A1A] pt-4">
                    <p className="text-[10px] uppercase tracking-wider text-stone-500 mb-2">Churn vs Retained Distribution</p>
                    <div className="flex h-3 rounded-full overflow-hidden text-[9px] text-center font-bold text-black font-mono">
                      <div style={{ width: `${100 - selectedDataset.churnRate}%` }} className="bg-stone-500 flex items-center justify-center text-white text-[8px]">
                        RETAINED: {100 - selectedDataset.churnRate}%
                      </div>
                      <div style={{ width: `${selectedDataset.churnRate}%` }} className="bg-[#C5B48C] flex items-center justify-center text-black text-[8px]">
                        CHURN: {selectedDataset.churnRate}%
                      </div>
                    </div>
                  </div>

                </div>

              </div>

              {/* SECOND ROW INTEL: COEF IMPORTANCES AND CORRELATION ANALYSIS */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                
                {/* Left side: Feature Importance bar list */}
                <div className="lg:col-span-6 p-6 border border-[#2A2A2A] bg-[#0A0A0A] rounded-sm">
                  <h4 className="text-[10px] uppercase tracking-[0.2em] text-[#555555] mb-4 font-bold">
                    Relative Gini Predictor Importance (Top 5)
                  </h4>
                  {metrics && (
                    <div className="space-y-4">
                      {metrics.featureImportance.map((f, idx) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="text-stone-300 font-medium">{f.feature}</span>
                            <span className="font-mono text-[#C5B48C] text-xs">{f.importance.toFixed(2)}</span>
                          </div>
                          <div className="w-full h-1.5 bg-[#121212] rounded-full overflow-hidden border border-[#222]">
                            <div 
                              className="bg-gradient-to-r from-[#8E7D5B] to-[#C5B48C] h-full rounded-full transition-all duration-1000"
                              style={{ width: `${f.importance * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  <p className="text-[9px] text-stone-500 mt-4">
                    Weights represent the relative Gini importance computed across all splits in our forest. Gini importance indicates the frequency of features utilized for tree segmentation.
                  </p>
                </div>

                {/* Right side: Correlation Analysis Table */}
                <div className="lg:col-span-6 p-6 border border-[#2A2A2A] bg-[#0D0D0D] rounded-sm">
                  <h4 className="text-[10px] uppercase tracking-[0.2em] text-[#555555] mb-4 font-bold">
                    Raw Pearson Correlation Coefficients (with Churn label)
                  </h4>
                  <div className="border border-[#2A2A2A] rounded overflow-hidden">
                    <table className="w-full text-xs">
                      <thead className="bg-[#121212] border-b border-[#2A2A2A] text-[10px] uppercase tracking-widest text-stone-500 font-semibold">
                        <tr>
                          <th className="p-2.5 text-left">Independent Feature</th>
                          <th className="p-2.5 text-right">Pearson Coefficient</th>
                          <th className="p-2.5 text-right">Association</th>
                        </tr>
                      </thead>
                      <tbody className="font-mono">
                        {selectedDataset.correlationData.slice(0, 5).map((c, i) => (
                          <tr key={i} className="border-b border-[#1A1A1A] hover:bg-[#121212]/50">
                            <td className="p-2.5 text-stone-400 text-left font-sans">{c.feature}</td>
                            <td className={`p-2.5 text-right font-semibold ${c.correlation > 0 ? 'text-red-400' : 'text-green-400'}`}>
                              {c.correlation > 0 ? `+${c.correlation.toFixed(2)}` : c.correlation.toFixed(2)}
                            </td>
                            <td className="p-2.5 text-right text-stone-500 font-sans text-[10px]">
                              {c.correlation > 0.3 ? 'Strong Risk Catalyst' : c.correlation < -0.2 ? 'Strong Retention Buffer' : 'Moderate'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  
                  {/* Action Link to Code Tab */}
                  <div className="mt-5 flex justify-between items-center bg-[#121212] p-3 rounded border border-[#222]">
                    <span className="text-[10px] text-stone-400">Want to generate the exact python code to train this?</span>
                    <button 
                      onClick={() => askAIPythonTemplate('pipeline')}
                      className="text-[10px] text-[#C5B48C] font-bold uppercase tracking-widest flex items-center gap-1 hover:underline"
                    >
                      Generate Python Code <ChevronRight className="w-3 h-3" />
                    </button>
                  </div>

                </div>

              </div>

            </div>
          </>
        )}

        {/* ==================================== TAB 2: CLIENT SIDE CSV TRAINER ==================================== */}
        {activeTab === 'csv-trainer' && (
          <div className="col-span-12 bg-[#0A0A0A] p-6 lg:p-10 flex flex-col overflow-y-auto">
            
            {/* Header Block */}
            <div className="mb-8 border-b border-[#1A1A1A] pb-6 flex flex-col lg:flex-row justify-between lg:items-end gap-4">
              <div>
                <p className="text-[10px] uppercase tracking-[0.3em] text-[#555555] mb-2">Advanced Offline Feature Pipeline</p>
                <h2 className="text-3xl lg:text-4xl font-serif italic text-white">
                  Client-Side <span className="text-[#C5B48C]">CSV ML Engine</span>
                </h2>
                <p className="text-xs text-stone-400 max-w-xl mt-1">
                  Upload, profile, and train a real binary Logistic Regression classifier directly inside your browser cache. No data ever leaves your device — secure, instant client-side machine learning!
                </p>
              </div>
              
              {/* Seed Dataset download shortcut */}
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    const csvContent = "tenure,MonthlyCharges,Contract,TechSupport,Churn\n12,45.5,month-to-month,no,Yes\n36,89.9,one-year,yes,No\n2,25.0,month-to-month,no,Yes\n48,110.0,two-year,yes,No\n6,65.0,month-to-month,no,Yes\n24,85.0,month-to-month,yes,No\n5,40.0,month-to-month,no,Yes\n60,115.0,two-year,yes,No\n8,50.0,month-to-month,no,Yes\n40,95.0,one-year,no,No\n1,35.0,month-to-month,no,Yes\n72,120.0,two-year,yes,No\n18,70.0,one-year,yes,No\n4,42.0,month-to-month,no,Yes\n30,80.0,month-to-month,no,Yes\n55,105.0,two-year,yes,No\n10,55.0,month-to-month,no,Yes\n50,90.0,one-year,no,No\n3,30.0,month-to-month,no,Yes\n64,112.0,two-year,yes,No";
                    const blob = new Blob([csvContent], { type: 'text/csv' });
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = 'customer_churn_sample.csv';
                    link.click();
                  }}
                  className="px-4 py-2 bg-[#121212] hover:bg-[#1C1C1C] text-stone-300 text-xs font-semibold rounded border border-[#2A2A2A] flex items-center gap-2"
                >
                  <Download className="w-3.5 h-3.5 text-[#C5B48C]" /> Get Sample CSV
                </button>
              </div>
            </div>

            {/* DRAG AND DROP ZONE */}
            {!csvPreview ? (
              <div 
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                className="border-2 border-dashed border-[#2A2A2A] hover:border-[#C5B48C] bg-[#0E0E0E] p-12 text-center rounded-lg flex flex-col items-center justify-center gap-4 transition-all"
              >
                <div className="w-16 h-16 bg-[#161616] rounded-full flex items-center justify-center border border-[#2A2A2A]">
                  <Upload className="w-8 h-8 text-[#C5B48C]" />
                </div>
                <div>
                  <p className="text-sm text-stone-200 font-semibold">Drag and drop your Customer Churn CSV here</p>
                  <p className="text-xs text-stone-500 mt-1">Accepts any raw tabular CSV containing demographic/billing columns and target churn labels</p>
                </div>
                <div className="relative">
                  <input 
                    type="file" 
                    accept=".csv" 
                    onChange={handleFileChange} 
                    className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                  />
                  <button className="px-6 py-2.5 bg-[#C5B48C] hover:opacity-90 text-black text-xs font-semibold uppercase tracking-wider rounded-sm">
                    Select File From Device
                  </button>
                </div>
              </div>
            ) : (
              /* ACTIVE TRAINING WORKSPACE */
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                
                {/* Left Side: File Stats and training hyperparameters */}
                <div className="lg:col-span-4 space-y-6">
                  
                  {/* CSV Stats Panel */}
                  <div className="p-5 border border-[#2A2A2A] bg-[#121212]/50 rounded-sm space-y-4">
                    <div className="flex justify-between items-center border-b border-[#2A2A2A] pb-3">
                      <div>
                        <span className="text-[9px] uppercase tracking-wider text-[#C5B48C] font-semibold">ACTIVE DATASET</span>
                        <h4 className="text-sm text-white font-mono">{csvFile?.name}</h4>
                      </div>
                      <button 
                        onClick={() => { setCsvFile(null); setCsvPreview(null); setCsvMetrics(null); }}
                        className="text-[10px] text-red-400 hover:underline"
                      >
                        Reset / Remove
                      </button>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div>
                        <span className="text-stone-500 block text-[10px]">Row Count</span>
                        <span className="font-mono text-white text-sm font-semibold">{csvPreview.rows.length}</span>
                      </div>
                      <div>
                        <span className="text-stone-500 block text-[10px]">Columns</span>
                        <span className="font-mono text-white text-sm font-semibold">{csvPreview.headers.length}</span>
                      </div>
                    </div>

                    {/* Choose Target Column */}
                    <div>
                      <label className="text-[11px] text-stone-400 block mb-1">Target / Output Column (y)</label>
                      <select
                        value={targetColumn}
                        onChange={(e) => {
                          setTargetColumn(e.target.value);
                          if (csvPreview) {
                            const profile = ClientSideMLEngine.profileDataset(csvPreview.headers, csvPreview.rows, e.target.value);
                            setCsvProfile(profile);
                          }
                        }}
                        className="w-full p-2.5 border border-[#2A2A2A] rounded bg-[#0A0A0A] text-xs font-mono text-[#C5B48C]"
                      >
                        {csvPreview.headers.map(h => (
                          <option key={h} value={h}>{h}</option>
                        ))}
                      </select>
                      <span className="text-[9px] text-stone-500 mt-1 block">Ensure the selected target column contains exactly two unique values (binary).</span>
                    </div>

                    {/* SGD Parameters */}
                    <div className="border-t border-[#2A2A2A] pt-4 space-y-3">
                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-stone-400">Gradient Step (Learning Rate)</span>
                          <span className="font-mono text-[#C5B48C]">{csvLr}</span>
                        </div>
                        <input 
                          type="range" 
                          min="0.01" 
                          max="0.5" 
                          step="0.01" 
                          value={csvLr}
                          onChange={(e) => setCsvLr(parseFloat(e.target.value))}
                          className="w-full accent-[#C5B48C] bg-[#222] h-1 rounded"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-stone-400">Max Training Epochs</span>
                          <span className="font-mono text-[#C5B48C]">{csvEpochs}</span>
                        </div>
                        <input 
                          type="range" 
                          min="10" 
                          max="150" 
                          step="5" 
                          value={csvEpochs}
                          onChange={(e) => setCsvEpochs(parseInt(e.target.value))}
                          className="w-full accent-[#C5B48C] bg-[#222] h-1 rounded"
                        />
                      </div>
                    </div>

                    <button
                      onClick={handleClientSideTrain}
                      disabled={isCsvTraining}
                      className="w-full py-3 bg-[#C5B48C] hover:opacity-90 active:scale-95 text-black font-semibold uppercase tracking-wider text-xs rounded-sm transition-all flex items-center justify-center gap-2 cursor-pointer"
                    >
                      {isCsvTraining ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          Training SGD Classifier...
                        </>
                      ) : (
                        <>
                          <Terminal className="w-3.5 h-3.5" />
                          Fit Logistic Regression
                        </>
                      )}
                    </button>

                  </div>

                  {/* Schema Profiler Panel */}
                  <div className="p-5 border border-[#2A2A2A] bg-[#121212]/50 rounded-sm">
                    <h4 className="text-[10px] uppercase tracking-[0.2em] text-[#555555] mb-3 font-bold">CSV Columns Profiler</h4>
                    <div className="max-h-[220px] overflow-y-auto space-y-2 pr-1">
                      {csvPreview.headers.map(h => (
                        <div key={h} className="p-2 bg-black border border-[#1A1A1A] rounded flex justify-between items-center text-[11px]">
                          <div>
                            <span className="font-mono text-stone-300 block max-w-[150px] truncate">{h}</span>
                            <span className="text-[9px] text-[#555] uppercase font-semibold">
                              {csvProfile?.[h]?.type || 'Categorical'}
                            </span>
                          </div>
                          <div className="text-right text-[10px] font-mono text-stone-500">
                            <div>Uniques: <span className="text-stone-300">{csvProfile?.[h]?.uniques ?? 'N/A'}</span></div>
                            <div>Miss: <span className="text-stone-300">{csvProfile?.[h]?.missing ?? 0}</span></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                </div>

                {/* Right Side: Training logs, results and live metrics */}
                <div className="lg:col-span-8 flex flex-col justify-between min-h-[400px]">
                  
                  {csvTrainingError && (
                    <div className="p-4 border border-red-950 bg-red-950/20 text-red-400 rounded-sm text-xs mb-4 flex items-start gap-3">
                      <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-bold">Execution Error during Stochastic Gradient Descent:</span>
                        <p className="mt-1">{csvTrainingError}</p>
                      </div>
                    </div>
                  )}

                  {!csvMetrics ? (
                    <div className="flex-1 border border-[#2A2A2A] bg-[#0E0E0E] rounded-sm p-10 flex flex-col items-center justify-center text-center">
                      <Terminal className="w-10 h-10 text-stone-700 mb-3" />
                      <p className="text-xs text-stone-400">Ready to train scikit-style SGD classifier.</p>
                      <p className="text-[10px] text-stone-600 max-w-sm mt-1">Click "Fit Logistic Regression" to calculate coefficients, scale numeric factors, parse categoricals, and evaluate performance indicators.</p>
                    </div>
                  ) : (
                    /* TRAINING METRICS & LOSS D3 PLOT */
                    <div className="flex-1 space-y-6">
                      
                      {/* KPI ROW */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                        {[
                          { label: 'Out-Of-Sample Accuracy', value: csvMetrics.accuracy },
                          { label: 'Trained Precision', value: csvMetrics.precision },
                          { label: 'Trained Recall', value: csvMetrics.recall },
                          { label: 'Trained ROC-AUC', value: csvMetrics.rocAuc }
                        ].map((metric, i) => (
                          <div key={i} className="p-4 bg-[#121212] border border-[#222] rounded">
                            <span className="text-[9px] uppercase tracking-widest text-[#555] block mb-1">{metric.label}</span>
                            <span className="text-2xl font-light font-mono text-white">{metric.value.toFixed(3)}</span>
                          </div>
                        ))}
                      </div>

                      {/* Training vs Val Loss plot side-by-side with Feature Weights */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        
                        <div className="p-5 border border-[#2A2A2A] bg-black rounded">
                          <h4 className="text-[10px] uppercase tracking-[0.2em] text-[#555555] mb-4 font-bold">Client SGD Loss Reduction Curve</h4>
                          <div className="h-[180px] w-full bg-[#0E0E0E] relative p-2 border border-[#1A1A1A]">
                            <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible" preserveAspectRatio="none">
                              <polyline
                                fill="none"
                                stroke="#888888"
                                strokeWidth="0.5"
                                strokeDasharray="1 1"
                                points={csvMetrics.lossCurve.map((lc, idx) => {
                                  const x = (idx / (csvMetrics.lossCurve.length - 1)) * 100;
                                  const y = 100 - (lc.loss / 1.5) * 100; // Cap scale to 1.5
                                  return `${x},${y}`;
                                }).join(' ')}
                              />
                              <polyline
                                fill="none"
                                stroke="#C5B48C"
                                strokeWidth="1.5"
                                points={csvMetrics.lossCurve.map((lc, idx) => {
                                  const x = (idx / (csvMetrics.lossCurve.length - 1)) * 100;
                                  const y = 100 - (lc.valLoss / 1.5) * 100;
                                  return `${x},${y}`;
                                }).join(' ')}
                              />
                            </svg>
                            <div className="absolute right-2 top-2 flex flex-col gap-1 text-[8px] font-mono text-stone-500">
                              <div><span className="w-2.5 h-0.5 bg-stone-500 inline-block"></span> Train Loss</div>
                              <div><span className="w-2.5 h-0.5 bg-[#C5B48C] inline-block"></span> Val Loss (20%)</div>
                            </div>
                          </div>
                          <p className="text-[8px] text-stone-500 mt-2 text-center font-mono">
                            Convergence achieved at Epoch {csvMetrics.lossCurve.length} (SGD Intercept Optimization).
                          </p>
                        </div>

                        <div className="p-5 border border-[#2A2A2A] bg-black rounded">
                          <h4 className="text-[10px] uppercase tracking-[0.2em] text-[#555555] mb-4 font-bold">Calculated Feature Weights</h4>
                          <div className="space-y-3.5">
                            {csvMetrics.featureImportance.slice(0, 5).map((f, i) => (
                              <div key={i} className="space-y-1">
                                <div className="flex justify-between text-xs font-mono">
                                  <span className="text-stone-400 max-w-[180px] truncate">{f.feature}</span>
                                  <span className="text-[#C5B48C]">{f.importance.toFixed(2)}</span>
                                </div>
                                <div className="w-full bg-[#121212] h-1.5 rounded-full overflow-hidden border border-[#222]">
                                  <div 
                                    className="bg-gradient-to-r from-[#8E7D5B] to-[#C5B48C] h-full"
                                    style={{ width: `${Math.min(100, f.importance * 100)}%` }}
                                  ></div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                      </div>

                    </div>
                  )}

                  {/* Bottom advice helper */}
                  <div className="mt-6 p-4 bg-[#121212] rounded border border-[#2A2A2A] text-xs text-stone-400">
                    <p className="font-serif italic text-[#C5B48C] mb-1.5">Note on client-side math execution:</p>
                    This is executing genuine standard stochastic gradient descent algorithms with min-max categorical transformations on your local thread in WebAssembly/TypeScript. Perfect for immediate prototyping before taking your model pipelines into full production scripts.
                  </div>

                </div>

              </div>
            )}

          </div>
        )}

        {/* ==================================== TAB 3: SINGLE CUSTOMER PREDICTION ==================================== */}
        {activeTab === 'single-pred' && (
          <div className="col-span-12 bg-[#0A0A0A] p-6 lg:p-10 flex flex-col overflow-y-auto">
            
            <div className="mb-8 border-b border-[#1A1A1A] pb-6">
              <p className="text-[10px] uppercase tracking-[0.3em] text-[#555555] mb-2">Simulate Real-time Customer Risk Profiles</p>
              <h2 className="text-3xl lg:text-4xl font-serif italic text-white">
                Individual <span className="text-[#C5B48C]">Predictive Calculator</span>
              </h2>
              <p className="text-xs text-stone-400 max-w-xl mt-1">
                Toggle configuration values below to see how a trained Logistic Regression estimator calculates an individual customer's attrition probability in real-time.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              
              {/* Left Side: Interactive Input Parameters */}
              <div className="lg:col-span-7 bg-[#0E0E0E] p-6 rounded border border-[#2A2A2A] space-y-6">
                
                <h3 className="text-xs uppercase tracking-[0.2em] text-[#C5B48C] font-semibold flex items-center gap-2">
                  <Sliders className="w-4 h-4" /> Demographics & Service Configuration
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                  
                  {/* Tenure Months */}
                  <div>
                    <label className="text-stone-400 block mb-1">Customer Tenure: <span className="font-mono text-[#C5B48C] font-bold">{customerInput.tenure} Months</span></label>
                    <input 
                      type="range" 
                      min="1" 
                      max="72" 
                      value={customerInput.tenure}
                      onChange={(e) => setCustomerInput(prev => ({ ...prev, tenure: parseInt(e.target.value) }))}
                      className="w-full accent-[#C5B48C] bg-[#222] h-1.5 rounded"
                    />
                    <span className="text-[9px] text-stone-500 mt-1 block">Longer tenure indicates deeper brand relationship.</span>
                  </div>

                  {/* Contract Style */}
                  <div>
                    <label className="text-stone-400 block mb-1">Subscription Contract Type</label>
                    <select
                      value={customerInput.contract}
                      onChange={(e) => setCustomerInput(prev => ({ ...prev, contract: e.target.value as any }))}
                      className="w-full p-2.5 border border-[#2A2A2A] bg-[#0A0A0A] text-xs rounded text-stone-300"
                    >
                      <option value="month-to-month">Month-to-Month (Flexible)</option>
                      <option value="one-year">One Year (Committed)</option>
                      <option value="two-year">Two Year (Highly Committed)</option>
                    </select>
                  </div>

                  {/* Monthly Charges */}
                  <div>
                    <label className="text-stone-400 block mb-1">Monthly Charges: <span className="font-mono text-[#C5B48C] font-bold">${customerInput.monthlyCharges}</span></label>
                    <input 
                      type="range" 
                      min="15" 
                      max="125" 
                      value={customerInput.monthlyCharges}
                      onChange={(e) => setCustomerInput(prev => ({ ...prev, monthlyCharges: parseInt(e.target.value) }))}
                      className="w-full accent-[#C5B48C] bg-[#222] h-1.5 rounded"
                    />
                  </div>

                  {/* Internet Service */}
                  <div>
                    <label className="text-stone-400 block mb-1">Internet Connection Style</label>
                    <select
                      value={customerInput.internetService}
                      onChange={(e) => setCustomerInput(prev => ({ ...prev, internetService: e.target.value as any }))}
                      className="w-full p-2.5 border border-[#2A2A2A] bg-[#0A0A0A] text-xs rounded text-stone-300"
                    >
                      <option value="fiber_optic">Fiber Optic (Premium/High Churn rate)</option>
                      <option value="dsl">DSL Broadband</option>
                      <option value="no">No Internet Package</option>
                    </select>
                  </div>

                  {/* Tech Support */}
                  <div>
                    <label className="text-stone-400 block mb-1">Tech Support Package</label>
                    <select
                      value={customerInput.techSupport}
                      onChange={(e) => setCustomerInput(prev => ({ ...prev, techSupport: e.target.value as any }))}
                      className="w-full p-2.5 border border-[#2A2A2A] bg-[#0A0A0A] text-xs rounded text-stone-300"
                    >
                      <option value="no">No Tech Support Subscribed</option>
                      <option value="yes">Premium Tech Support (Yes)</option>
                      <option value="no_internet">No Internet Connection Available</option>
                    </select>
                  </div>

                  {/* Payment Method */}
                  <div>
                    <label className="text-stone-400 block mb-1">Payment Style</label>
                    <select
                      value={customerInput.paymentMethod}
                      onChange={(e) => setCustomerInput(prev => ({ ...prev, paymentMethod: e.target.value as any }))}
                      className="w-full p-2.5 border border-[#2A2A2A] bg-[#0A0A0A] text-xs rounded text-stone-300"
                    >
                      <option value="electronic_check">Electronic Check (Manual)</option>
                      <option value="mailed_check">Mailed Paper Check</option>
                      <option value="bank_transfer">Bank Transfer (Auto-pay)</option>
                      <option value="credit_card">Credit Card (Auto-pay)</option>
                    </select>
                  </div>

                  {/* Demographics row - toggles */}
                  <div className="md:col-span-2 grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-[#1C1C1C]">
                    
                    <div>
                      <span className="text-stone-500 block text-[10px] mb-1">Paperless Bill</span>
                      <div className="flex gap-1">
                        {['yes', 'no'].map(opt => (
                          <button
                            key={opt}
                            onClick={() => setCustomerInput(prev => ({ ...prev, paperlessBilling: opt as any }))}
                            className={`flex-1 py-1 text-[10px] font-mono uppercase rounded text-center border ${
                              customerInput.paperlessBilling === opt 
                                ? 'border-[#C5B48C] bg-[#C5B48C]/10 text-[#C5B48C]' 
                                : 'border-[#2A2A2A] bg-black text-stone-600'
                            }`}
                          >
                            {opt}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div>
                      <span className="text-stone-500 block text-[10px] mb-1">Senior Citizen</span>
                      <div className="flex gap-1">
                        {['yes', 'no'].map(opt => (
                          <button
                            key={opt}
                            onClick={() => setCustomerInput(prev => ({ ...prev, seniorCitizen: opt as any }))}
                            className={`flex-1 py-1 text-[10px] font-mono uppercase rounded text-center border ${
                              customerInput.seniorCitizen === opt 
                                ? 'border-[#C5B48C] bg-[#C5B48C]/10 text-[#C5B48C]' 
                                : 'border-[#2A2A2A] bg-black text-stone-600'
                            }`}
                          >
                            {opt}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div>
                      <span className="text-stone-500 block text-[10px] mb-1">Has Partner</span>
                      <div className="flex gap-1">
                        {['yes', 'no'].map(opt => (
                          <button
                            key={opt}
                            onClick={() => setCustomerInput(prev => ({ ...prev, partner: opt as any }))}
                            className={`flex-1 py-1 text-[10px] font-mono uppercase rounded text-center border ${
                              customerInput.partner === opt 
                                ? 'border-[#C5B48C] bg-[#C5B48C]/10 text-[#C5B48C]' 
                                : 'border-[#2A2A2A] bg-black text-stone-600'
                            }`}
                          >
                            {opt}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div>
                      <span className="text-stone-500 block text-[10px] mb-1">Dependents</span>
                      <div className="flex gap-1">
                        {['yes', 'no'].map(opt => (
                          <button
                            key={opt}
                            onClick={() => setCustomerInput(prev => ({ ...prev, dependents: opt as any }))}
                            className={`flex-1 py-1 text-[10px] font-mono uppercase rounded text-center border ${
                              customerInput.dependents === opt 
                                ? 'border-[#C5B48C] bg-[#C5B48C]/10 text-[#C5B48C]' 
                                : 'border-[#2A2A2A] bg-black text-stone-600'
                            }`}
                          >
                            {opt}
                          </button>
                        ))}
                      </div>
                    </div>

                  </div>

                </div>

              </div>

              {/* Right Side: Risk Probability Ring and Driver analysis */}
              <div className="lg:col-span-5 flex flex-col gap-6">
                
                {/* Risk Ring Screen */}
                {predictionOutcome && (
                  <div className="p-8 border border-[#2A2A2A] bg-gradient-to-br from-[#121212] to-[#0A0A0A] rounded-sm text-center relative overflow-hidden flex flex-col items-center">
                    
                    <span className="text-[10px] uppercase tracking-[0.2em] text-[#888888] mb-1">ESTIMATED CHURN RISK</span>
                    <h3 className={`text-sm font-serif italic mb-6 ${
                      predictionOutcome.riskLevel === 'High' ? 'text-red-400' : predictionOutcome.riskLevel === 'Medium' ? 'text-orange-400' : 'text-green-400'
                    }`}>
                      {predictionOutcome.riskLevel} Churn Risk
                    </h3>

                    {/* SVG Radial Gauge */}
                    <div className="relative w-40 h-40 flex items-center justify-center mb-6">
                      <svg className="w-full h-full transform -rotate-90">
                        <circle
                          cx="80"
                          cy="80"
                          r="68"
                          className="stroke-stone-900"
                          strokeWidth="8"
                          fill="transparent"
                        />
                        <circle
                          cx="80"
                          cy="80"
                          r="68"
                          className={`transition-all duration-700 ${
                            predictionOutcome.riskLevel === 'High' ? 'stroke-red-400' : predictionOutcome.riskLevel === 'Medium' ? 'stroke-orange-400' : 'stroke-[#C5B48C]'
                          }`}
                          strokeWidth="8"
                          strokeDasharray={2 * Math.PI * 68}
                          strokeDashoffset={2 * Math.PI * 68 * (1 - predictionOutcome.probability / 100)}
                          fill="transparent"
                          strokeLinecap="round"
                        />
                      </svg>
                      <div className="absolute flex flex-col items-center">
                        <span className="text-4xl font-light text-white font-mono">{predictionOutcome.probability}%</span>
                        <span className="text-[10px] text-stone-500 uppercase tracking-widest mt-1">Probability</span>
                      </div>
                    </div>

                    <div className="w-full text-left space-y-3 pt-6 border-t border-[#2A2A2A]">
                      <span className="text-[9px] uppercase tracking-widest text-[#555] block">Key Risk Drivers / Retention Safeguards:</span>
                      <div className="space-y-2">
                        {predictionOutcome.factors.map((f: string, i: number) => {
                          const isRisk = f.includes('Risk') || f.includes('increases') || f.includes('friction');
                          return (
                            <div key={i} className="flex gap-2.5 text-xs text-stone-400 leading-relaxed">
                              {isRisk ? (
                                <AlertTriangle className="w-3.5 h-3.5 text-red-400 shrink-0 mt-0.5" />
                              ) : (
                                <CheckCircle className="w-3.5 h-3.5 text-green-400 shrink-0 mt-0.5" />
                              )}
                              <span>{f}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                  </div>
                )}

                {/* Simulated CRM actions advice */}
                <div className="p-6 border border-[#2A2A2A] bg-[#0E0E0E] rounded">
                  <h4 className="text-[11px] uppercase tracking-widest text-[#C5B48C] mb-2 font-bold">Recommended CRM Mitigation Action</h4>
                  <p className="text-xs text-stone-400 leading-relaxed">
                    {predictionOutcome?.probability > 65 ? (
                      "Critical: Retain this user immediately. Push a contract migration offer (convert to 1-Year with 15% discount) to establish legal retention lock. Dispatch customer success agent to check Fiber configuration setup issues."
                    ) : predictionOutcome?.probability > 35 ? (
                      "Standard: High monthly charge exposure detected. Provide standard promotional loyalty bundle credit ($10 discount on next invoice) or tech support health check package."
                    ) : (
                      "Low Friction: High retention health score. Standard retention schedule. Recommend upselling additional products or cross-promoting family plans."
                    )}
                  </p>
                </div>

              </div>

            </div>

          </div>
        )}

        {/* ==================================== TAB 4: CHAT WITH AI CONSULTANT ==================================== */}
        {activeTab === 'ai-expert' && (
          <div className="col-span-12 bg-[#0A0A0A] flex flex-col md:flex-row flex-1 overflow-hidden min-h-[500px]">
            
            {/* Left Column: Fast code shortcuts */}
            <div className="w-full md:w-80 bg-[#0D0D0D] border-b md:border-b-0 md:border-r border-[#2A2A2A] p-6 shrink-0 flex flex-col justify-between">
              
              <div className="space-y-6">
                <div>
                  <h3 className="text-xs uppercase tracking-[0.2em] text-[#C5B48C] font-semibold mb-2">Python Snippet Library</h3>
                  <p className="text-[11px] text-stone-500 leading-relaxed">
                    Select a core machine learning topic below to populate code parameters into the chat context:
                  </p>
                </div>

                <div className="space-y-2">
                  <button
                    onClick={() => askAIPythonTemplate('pipeline')}
                    className="w-full text-left p-3 bg-[#121212] hover:bg-[#1A1A1A] border border-[#222] rounded text-xs text-stone-300 transition-all flex items-start gap-2.5"
                  >
                    <BookOpen className="w-4 h-4 text-[#C5B48C] shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold block text-stone-200">Preprocessing Pipeline</span>
                      <p className="text-[10px] text-stone-500 mt-1">Imputer, StandardScaler, and OneHotEncoder columns transformer.</p>
                    </div>
                  </button>

                  <button
                    onClick={() => askAIPythonTemplate('imbalance')}
                    className="w-full text-left p-3 bg-[#121212] hover:bg-[#1A1A1A] border border-[#222] rounded text-xs text-stone-300 transition-all flex items-start gap-2.5"
                  >
                    <Sliders className="w-4 h-4 text-[#C5B48C] shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold block text-stone-200">Handle Class Imbalance</span>
                      <p className="text-[10px] text-stone-500 mt-1">SMOTE oversampling with balanced validation strategies.</p>
                    </div>
                  </button>

                  <button
                    onClick={() => askAIPythonTemplate('boosting')}
                    className="w-full text-left p-3 bg-[#121212] hover:bg-[#1A1A1A] border border-[#222] rounded text-xs text-stone-300 transition-all flex items-start gap-2.5"
                  >
                    <Terminal className="w-4 h-4 text-[#C5B48C] shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold block text-stone-200">XGBoost & Optuna Tuning</span>
                      <p className="text-[10px] text-stone-500 mt-1">Hyperparameter search space optimizing ROC-AUC score.</p>
                    </div>
                  </button>
                </div>
              </div>

              <div className="p-4 bg-[#121212] rounded border border-[#222] mt-6 text-[10px] text-stone-500">
                <p className="font-serif italic text-[#C5B48C] mb-1">Consulting scope:</p>
                Answers are calibrated directly by state-of-the-art LLMs to draft executable Scikit-Learn code.
              </div>

            </div>

            {/* Right Column: Dynamic Gemini chat container */}
            <div className="flex-1 flex flex-col justify-between bg-[#0A0A0A]">
              
              {/* Chat Messages flow */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.map((m) => (
                  <div 
                    key={m.id} 
                    className={`flex gap-3 max-w-3xl ${m.role === 'user' ? 'ml-auto flex-row-reverse' : ''}`}
                  >
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center border text-xs shrink-0 ${
                      m.role === 'user' 
                        ? 'bg-stone-800 border-stone-700 text-white' 
                        : 'bg-[#1D1A13] border-[#C5B48C]/40 text-[#C5B48C]'
                    }`}>
                      {m.role === 'user' ? <User className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5" />}
                    </div>

                    <div className={`p-4 rounded text-xs leading-relaxed max-w-2xl overflow-x-auto ${
                      m.role === 'user' 
                        ? 'bg-[#1D1C1A] text-stone-200 rounded-tr-none border border-stone-800' 
                        : 'bg-[#121212] text-stone-300 rounded-tl-none border border-[#222]'
                    }`}>
                      {/* Formatted body text (handles raw markdown line splits) */}
                      <div className="whitespace-pre-wrap font-sans space-y-2">
                        {m.content.split('\n').map((line, idx) => {
                          if (line.startsWith('```')) return null; // Simple code block ignore wrapper (formatted cleanly as pre)
                          return <p key={idx}>{line}</p>;
                        })}
                      </div>
                    </div>
                  </div>
                ))}
                
                {isAiLoading && (
                  <div className="flex gap-3 max-w-xl">
                    <div className="w-8 h-8 rounded-full bg-[#1A1813] border border-[#C5B48C]/40 text-[#C5B48C] flex items-center justify-center shrink-0">
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    </div>
                    <div className="p-4 bg-[#121212] rounded text-xs text-stone-500 font-mono italic">
                      Computing predictive Python instructions...
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Form Box */}
              <form onSubmit={handleSendMessage} className="p-4 border-t border-[#2A2A2A] bg-[#0E0E0E] flex gap-3">
                <input
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Ask for Python templates, SMOTE imbalance handlers, evaluation plots..."
                  className="flex-1 px-4 py-3 bg-[#080808] border border-[#2A2A2A] rounded-sm text-xs font-mono text-white focus:outline-none focus:border-[#C5B48C] placeholder-stone-600"
                />
                <button
                  type="submit"
                  disabled={isAiLoading || !inputText.trim()}
                  className="px-6 py-3 bg-[#C5B48C] hover:opacity-90 disabled:opacity-50 text-black font-semibold uppercase tracking-wider text-xs rounded-sm transition-all flex items-center gap-2 cursor-pointer"
                >
                  <Send className="w-3.5 h-3.5" /> Send
                </button>
              </form>

            </div>

          </div>
        )}

      </main>

      {/* FOOTER DETAILS */}
      <footer id="workspace-footer" className="h-12 bg-[#080808] border-t border-[#2A2A2A] px-6 lg:px-10 flex items-center justify-between text-[10px] text-[#444444] uppercase tracking-[0.2em]">
        <div>Model Context Hash: <span className="font-mono">7fb921x...220</span></div>
        <div className="hidden sm:block">Customer Churn Simulator Workspace</div>
        <div className="flex gap-4">
          <span className="text-[#666666]">Isolation Enclave Safe</span>
          <span className="text-[#C5B48C]">SDK v2.4.0</span>
        </div>
      </footer>

    </div>
  );
}
