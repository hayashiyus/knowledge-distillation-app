import React, { useState, useEffect } from 'react';  // useEffectを追加
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

interface GenerateRequest {
  title: string;
  body: string;
  treatment: string;
  processing_type?: string;
  num_candidates: number;
  enable_learning: boolean;
  enable_web_search: boolean;
  json_data?: any;
  pdf_references?: string[];
  csv_references?: string[];
  user_profile?: any;
  user_history?: any[];
  use_single_model?: boolean;
  model_name?: string;
}

interface TournamentMatch {
  match: string;
  scores: string;
  winner: string;
}

interface TournamentRound {
  round: number | string;
  type: string;
  matches?: TournamentMatch[];
  model?: string;
  score?: number;
}

interface GenerateResponse {
  response: string;
  model_used: string;
  persuasion_score: number;
  tournament_log: TournamentRound[];
  processing_time: number;
  single_model_mode?: boolean;
}

interface PDFListResponse {
  pdfs: string[];
  full_paths: string[];
  directory: string;
  count: number;
  error?: string;
}

interface CSVListResponse {
  csvs: string[];
  full_paths: string[];
  count: number;
  error?: string;
}

function App() {
  const [title, setTitle] = useState('');
  const [body, setBody] = useState('');
  const [treatment, setTreatment] = useState('generic');
  const [numCandidates, setNumCandidates] = useState(4);
  const [enableWebSearch, setEnableWebSearch] = useState(false);
  const [enableLearning, setEnableLearning] = useState(true);
  const [jsonFile, setJsonFile] = useState<File | null>(null);
  const [jsonData, setJsonData] = useState<any>(null);
  const [response, setResponse] = useState<GenerateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useSingleModel, setUseSingleModel] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('gpt-4.1');
  const [availablePdfs, setAvailablePdfs] = useState<string[]>([]);
  const [selectedPdfs, setSelectedPdfs] = useState<string[]>([]);
  const [availableCsvs, setAvailableCsvs] = useState<string[]>([]);  // 追加
  const [selectedCsvs, setSelectedCsvs] = useState<string[]>([]);    // 追加

  // PDFファイルのリストを取得する関数
  const fetchAvailablePdfs = async () => {
    try {
      const response = await axios.get<PDFListResponse>(`${API_BASE}/api/available-pdfs`);
      if (response.data.pdfs && response.data.full_paths) {
        setAvailablePdfs(response.data.full_paths);
        console.log('Available PDFs:', response.data);
      }
    } catch (error) {
      console.error('Failed to fetch PDFs:', error);
    }
  };

  const fetchAvailableCsvs = async () => {
    try {
      const response = await axios.get<CSVListResponse>(`${API_BASE}/api/available-csvs`);
      if (response.data.csvs && response.data.full_paths) {
        setAvailableCsvs(response.data.full_paths);
        console.log('Available CSVs:', response.data);
      }
    } catch (error) {
      console.error('Failed to fetch CSVs:', error);
    }
  };

  // コンポーネントのマウント時にPDFリストを取得
  useEffect(() => {
    fetchAvailablePdfs();
    fetchAvailableCsvs();  // 追加
  }, []);

  const generateResponse = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const request: GenerateRequest = {
        title,
        body,
        treatment,
        processing_type: treatment,
        num_candidates: useSingleModel ? 1 : numCandidates,
        enable_learning: enableLearning && !useSingleModel,
        enable_web_search: enableWebSearch,
        json_data: jsonData,
        pdf_references: selectedPdfs,
        csv_references: selectedCsvs,
        use_single_model: useSingleModel,
        model_name: useSingleModel ? selectedModel : undefined
      };
      
      const result = await axios.post<GenerateResponse>(`${API_BASE}/api/generate`, request);
      
      setResponse(result.data);
    } catch (err: any) {
      let errorMessage = 'An error occurred';
      if (err.response?.data?.detail) {
        if (typeof err.response.data.detail === 'string') {
          errorMessage = err.response.data.detail;
        } else if (typeof err.response.data.detail === 'object') {
          errorMessage = err.response.data.detail.msg || 
                        err.response.data.detail.message || 
                        JSON.stringify(err.response.data.detail);
        }
      } else if (err.message) {
        errorMessage = err.message;
      }
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>意思決定支援のための応答生成システム</h1>
        <p>複数のAIを使用して情報を融合し意思決定支援のための応答を生成します</p>
      </header>

      <main className="App-main">
        <div className="input-section">
          <h2>質問を入力</h2>
          
          <div className="form-group">
            <label htmlFor="title">タイトル</label>
            <input
              id="title"
              type="text"
              placeholder="例: CMV: 参政党の躍進は既存政党の問題を示している"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="title-input"
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="body">本文</label>
            <textarea
              id="body"
              placeholder="あなたの意見や議論を詳しく記述してください..."
              value={body}
              onChange={(e) => setBody(e.target.value)}
              rows={8}
              className="body-input"
            />
          </div>
          
          <div className="options">
            <div className="form-group">
              <label htmlFor="treatment">処理タイプ:</label>
              <select 
                id="treatment"
                value={treatment} 
                onChange={(e) => setTreatment(e.target.value)}
              >
                <option value="generic">汎用</option>
                <option value="personalization">パーソナライゼーション</option>
                <option value="community_aligned">コミュニティ最適化</option>
              </select>
            </div>
            
            <div className="form-group">
              <label htmlFor="candidates">候補数: {numCandidates}</label>
              <input
                id="candidates"
                type="range"
                min="2"
                max="128"
                value={numCandidates}
                onChange={(e) => setNumCandidates(Number(e.target.value))}
                disabled={useSingleModel}
              />
            </div>
            
            <div className="form-group checkbox">
              <label>
                <input
                  type="checkbox"
                  checked={enableWebSearch}
                  onChange={(e) => setEnableWebSearch(e.target.checked)}
                />
                Web検索を有効にする
              </label>
            </div>
            
            <div className="form-group checkbox">
              <label>
                <input
                  type="checkbox"
                  checked={enableLearning}
                  onChange={(e) => setEnableLearning(e.target.checked)}
                  disabled={useSingleModel}
                />
                学習機能を有効にする
              </label>
            </div>
            
            <div className="form-group">
              <label htmlFor="json-upload">JSONファイル（YouTube/TikTokデータ）</label>
              <input
                id="json-upload"
                type="file"
                accept=".json"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    setJsonFile(file);
                    const reader = new FileReader();
                    reader.onload = (event) => {
                      try {
                        const data = JSON.parse(event.target?.result as string);
                        setJsonData(data);
                        console.log('JSON data loaded:', data);
                      } catch (error) {
                        console.error('Invalid JSON file:', error);
                        setError('無効なJSONファイルです');
                      }
                    };
                    reader.readAsText(file);
                  }
                }}
              />
              {jsonFile && (
                <small className="file-info">
                  ファイル: {jsonFile.name} ({(jsonFile.size / 1024).toFixed(2)} KB)
                </small>
              )}
            </div>

            {/* PDFファイル選択（利用可能な場合） */}
            {availablePdfs.length > 0 && (
              <div className="form-group">
                <label>PDFファイルを選択（複数選択可）</label>
                <div className="pdf-list" style={{ maxHeight: '150px', overflowY: 'auto', border: '1px solid #ddd', padding: '10px', borderRadius: '4px' }}>
                  {availablePdfs.map((pdfPath, index) => (
                    <label key={index} className="checkbox" style={{ display: 'block', marginBottom: '5px' }}>
                      <input
                        type="checkbox"
                        value={pdfPath}
                        checked={selectedPdfs.includes(pdfPath)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedPdfs([...selectedPdfs, pdfPath]);
                          } else {
                            setSelectedPdfs(selectedPdfs.filter(p => p !== pdfPath));
                          }
                        }}
                      />
                      <span style={{ marginLeft: '5px' }}>{pdfPath.split('/').pop()}</span>
                    </label>
                  ))}
                </div>
                {selectedPdfs.length > 0 && (
                  <small className="file-info">
                    選択済み: {selectedPdfs.length}件
                  </small>
                )}
              </div>
            )}

            {/* CSVファイル選択セクション（新規追加） */}
            {availableCsvs.length > 0 && (
              <div className="form-group">
                <label>CSVファイルを選択（X/Twitterデータ・複数選択可）</label>
                <div className="csv-list" style={{ 
                  maxHeight: '150px', 
                  overflowY: 'auto', 
                  border: '1px solid #ddd', 
                  padding: '10px', 
                  borderRadius: '4px',
                  backgroundColor: '#f8f9fa'
                }}>
                  {availableCsvs.map((csvPath, index) => (
                    <label key={index} className="checkbox" style={{ 
                      display: 'block', 
                      marginBottom: '5px',
                      cursor: 'pointer',
                      padding: '3px',
                      borderRadius: '3px',
                      transition: 'background-color 0.2s'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#e9ecef'}
                    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                    >
                      <input
                        type="checkbox"
                        value={csvPath}
                        checked={selectedCsvs.includes(csvPath)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedCsvs([...selectedCsvs, csvPath]);
                          } else {
                            setSelectedCsvs(selectedCsvs.filter(c => c !== csvPath));
                          }
                        }}
                      />
                      <span style={{ 
                        marginLeft: '8px',
                        fontSize: '14px',
                        color: selectedCsvs.includes(csvPath) ? '#0066cc' : '#333'
                      }}>
                        📊 {csvPath.split('/').pop()}
                      </span>
                    </label>
                  ))}
                </div>
                {selectedCsvs.length > 0 && (
                  <small className="file-info" style={{ color: '#0066cc', fontWeight: 'bold' }}>
                    選択済みCSV: {selectedCsvs.length}件
                  </small>
                )}
              </div>
            )}

            {/* 単一モデルモードの設定 */}
            <div className="form-group">
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={useSingleModel}
                  onChange={(e) => setUseSingleModel(e.target.checked)}
                />
                単一モデルモード
              </label>
              
              {useSingleModel && (
                <div className="model-select">
                  <label htmlFor="model-select">使用するモデル</label>
                  <select
                    id="model-select"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    <option value="gpt-4.1">GPT-4.1</option>
                    <option value="claude-sonnet-4">Claude Sonnet 4</option>
                    <option value="gemini-2.5-pro">Gemini 2.5 Pro</option>
                    <option value="llama-3.1-70b">Llama 3.3 70B (OpenRouter)</option>
                  </select>
                </div>
              )}
            </div>
          </div>
          
          <button 
            onClick={generateResponse} 
            disabled={loading || !title || !body}
            className="generate-btn"
          >
            {loading ? '生成中...' : '応答を生成'}
          </button>
        </div>

        {error && (
          <div className="error">
            エラー: {error}
          </div>
        )}

        {response && (
          <div className="response-section">
            <h2>生成された応答</h2>
            <div className="response-content">
              <p className="response-text">{response.response}</p>
              <div className="metadata">
                <span>モデル: {response.model_used}</span>
                <span>スコア: {response.persuasion_score.toFixed(2)}</span>
                <span>処理時間: {response.processing_time.toFixed(1)}秒</span>
                {jsonData && <span>JSONデータ: 使用済み</span>}
                {selectedPdfs.length > 0 && <span>PDF: {selectedPdfs.length}件使用</span>}
                {selectedCsvs.length > 0 && <span>CSV: {selectedCsvs.length}件使用</span>}  {/* 追加 */}
              </div>
            </div>

            {response.tournament_log && response.tournament_log.length > 0 && (
              <div className="tournament-section">
                <h3>
                  {response.single_model_mode ? 'モデル情報' : 'トーナメント結果'}
                </h3>
                
                {response.single_model_mode ? (
                  <div className="single-model-info">
                    <p>使用モデル: {response.model_used}</p>
                    <p>スコア: {response.persuasion_score.toFixed(2)}</p>
                  </div>
                ) : (
                  <div className="tournament-rounds">
                    {response.tournament_log.map((round, idx) => (
                      <div key={idx} className="round">
                        {round.type === 'initial' ? (
                          <h4>初期候補</h4>
                        ) : round.type === 'learning_summary' ? (
                          <div>
                            <h4>学習サマリー</h4>
                            <p>総改善回数: {(round as any).total_improvements || 0}</p>
                          </div>
                        ) : (
                          <>
                            <h4>ラウンド {round.round}</h4>
                            {round.matches?.map((match, midx) => (
                              <div key={midx} className="match">
                                <span className="match-info">{match.match}</span>
                                <span className="scores">スコア: {match.scores}</span>
                                <span className="winner">勝者: {match.winner}</span>
                              </div>
                            ))}
                          </>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;