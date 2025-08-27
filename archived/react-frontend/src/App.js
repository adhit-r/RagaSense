import React, { useState, useEffect } from 'react';
import { Upload, Volume2, Mic, Loader2, CheckCircle, XCircle } from 'lucide-react';
import axios from 'axios';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import RagaList from './pages/RagaList';

const API_URL = 'http://localhost:8000/api';

function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioElement] = useState(new Audio());

  const [audioAnalysis, setAudioAnalysis] = useState(null);
  const [audioAnalysisLoading, setAudioAnalysisLoading] = useState(false);
  const [audioAnalysisError, setAudioAnalysisError] = useState(null);

  const [ragaAnalysis, setRagaAnalysis] = useState(null);
  const [ragaAnalysisLoading, setRagaAnalysisLoading] = useState(false);
  const [ragaAnalysisError, setRagaAnalysisError] = useState(null);

  const [compareRagas, setCompareRagas] = useState([]);
  const [selectedCompareId, setSelectedCompareId] = useState('');
  const [compareResult, setCompareResult] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioFile(file);
      setAudioUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!audioFile) {
      setError('Please select an audio file first');
      return;
    }

    const formData = new FormData();
    formData.append('file', audioFile);

    try {
      setIsAnalyzing(true);
      setError(null);
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred during analysis');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const togglePlayback = () => {
    if (!audioUrl) return;

    if (isPlaying) {
      audioElement.pause();
      audioElement.currentTime = 0;
    } else {
      audioElement.src = audioUrl;
      audioElement.play();
    }
    setIsPlaying(!isPlaying);
  };

  // Fetch all ragas for comparison dropdown
  useEffect(() => {
    async function fetchRagas() {
      try {
        const res = await axios.get(`${API_URL}/ragas?limit=1000`);
        setCompareRagas(res.data);
      } catch (e) {
        // ignore for now
      }
    }
    fetchRagas();
  }, []);

  // After detection, fetch audio analysis and raga analysis
  useEffect(() => {
    if (result && audioFile) {
      // Audio analysis
      (async () => {
        setAudioAnalysisLoading(true);
        setAudioAnalysisError(null);
        try {
          const formData = new FormData();
          formData.append('file', audioFile);
          const res = await axios.post(`${API_URL}/analyze`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
          });
          setAudioAnalysis(res.data);
        } catch (e) {
          setAudioAnalysisError(e.response?.data?.detail || 'Audio analysis failed');
        } finally {
          setAudioAnalysisLoading(false);
        }
      })();
      // Raga analysis
      (async () => {
        setRagaAnalysisLoading(true);
        setRagaAnalysisError(null);
        try {
          // Assume result has raga_id or similar
          const ragaId = result.raga_id || result.raga_info?.id || result.raga_info?.raga_id;
          if (ragaId) {
            const res = await axios.get(`${API_URL}/analysis/raga/${ragaId}`);
            setRagaAnalysis(res.data);
          }
        } catch (e) {
          setRagaAnalysisError(e.response?.data?.detail || 'Raga analysis failed');
        } finally {
          setRagaAnalysisLoading(false);
        }
      })();
    } else {
      setAudioAnalysis(null);
      setRagaAnalysis(null);
    }
  }, [result, audioFile]);

  // Handle raga comparison
  const handleCompare = async () => {
    if (!result || !selectedCompareId) return;
    setCompareLoading(true);
    setCompareError(null);
    try {
      const ragaId = result.raga_id || result.raga_info?.id || result.raga_info?.raga_id;
      const res = await axios.get(`${API_URL}/ragas/compare`, {
        params: { id: [ragaId, selectedCompareId] },
      });
      setCompareResult(res.data);
    } catch (e) {
      setCompareError(e.response?.data?.detail || 'Comparison failed');
    } finally {
      setCompareLoading(false);
    }
  };

  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-4">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-5xl font-bold text-white mb-4 bg-gradient-to-r from-yellow-400 to-orange-500 bg-clip-text text-transparent">
              🎵 AI Raga Detection System
            </h1>
            <p className="text-gray-300 text-lg">
              Upload or record an audio clip to detect the raga
            </p>
            <nav className="mt-4 flex justify-center gap-6">
              <Link to="/" className="text-white hover:text-yellow-400 font-medium">Home</Link>
              <Link to="/ragas" className="text-white hover:text-yellow-400 font-medium">Raga List</Link>
            </nav>
          </div>
          <Routes>
            <Route path="/" element={
              <React.Fragment>
                {/* Upload Section */}
                <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                  <h2 className="text-2xl font-semibold text-white mb-6 flex items-center">
                    <Upload className="mr-3" />
                    Upload Audio
                  </h2>
                  
                  <div className="border-2 border-dashed border-white/20 rounded-xl p-8 text-center mb-6">
                    <input
                      type="file"
                      id="audio-upload"
                      className="hidden"
                      accept="audio/*"
                      onChange={handleFileChange}
                    />
                    <label
                      htmlFor="audio-upload"
                      className="cursor-pointer flex flex-col items-center justify-center space-y-4"
                    >
                      <div className="p-4 bg-white/10 rounded-full">
                        <Upload size={32} className="text-white" />
                      </div>
                      <div>
                        <p className="text-white font-medium">
                          {audioFile ? audioFile.name : 'Click to upload or drag and drop'}
                        </p>
                        <p className="text-sm text-gray-400 mt-1">
                          {audioFile ? 'Click to change file' : 'Supports WAV, MP3, FLAC, OGG, M4A'}
                        </p>
                      </div>
                    </label>
                  </div>

                  <div className="flex justify-between items-center">
                    <button
                      onClick={togglePlayback}
                      disabled={!audioUrl}
                      className={`flex items-center px-6 py-3 rounded-full font-medium transition-colors ${
                        audioUrl
                          ? 'bg-blue-600 hover:bg-blue-700 text-white'
                          : 'bg-gray-600 cursor-not-allowed text-gray-400'
                      }`}
                    >
                      {isPlaying ? (
                        <>
                          <XCircle className="mr-2" size={20} />
                          Stop
                        </>
                      ) : (
                        <>
                          <Volume2 className="mr-2" size={20} />
                          Play
                        </>
                      )}
                    </button>

                    <button
                      onClick={handleAnalyze}
                      disabled={!audioFile || isAnalyzing}
                      className={`flex items-center px-6 py-3 rounded-full font-medium transition-colors ${
                        audioFile && !isAnalyzing
                          ? 'bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600 text-white'
                          : 'bg-gray-600 cursor-not-allowed text-gray-400'
                      }`}
                    >
                      {isAnalyzing ? (
                        <>
                          <Loader2 className="animate-spin mr-2" size={20} />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Mic className="mr-2" size={20} />
                          Detect Raga
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Results Section */}
                <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                  <h2 className="text-2xl font-semibold text-white mb-6">Analysis Results</h2>
                  
                  {error && (
                    <div className="bg-red-500/20 border border-red-500/50 text-red-100 p-4 rounded-lg mb-6">
                      <p className="font-medium">Error</p>
                      <p className="text-sm">{error}</p>
                    </div>
                  )}

                  {isAnalyzing ? (
                    <div className="flex flex-col items-center justify-center py-12">
                      <Loader2 className="animate-spin text-yellow-400 mb-4" size={48} />
                      <p className="text-white font-medium">Analyzing your audio...</p>
                      <p className="text-gray-400 text-sm mt-2">This may take a few moments</p>
                    </div>
                  ) : result ? (
                    <div className="space-y-6">
                      <div className="bg-gradient-to-r from-yellow-500/20 to-orange-500/20 p-6 rounded-xl border border-yellow-500/30">
                        <div className="flex items-center justify-between">
                          <div>
                            <h3 className="text-3xl font-bold text-white">{result.predicted_raga}</h3>
                            <p className="text-yellow-400 mt-1">
                              Confidence: {(result.confidence[result.predicted_raga] * 100).toFixed(1)}%
                            </p>
                          </div>
                          <div className="bg-yellow-500/20 p-3 rounded-full">
                            <CheckCircle className="text-yellow-400" size={32} />
                          </div>
                        </div>
                      </div>

                      <div className="bg-white/5 p-6 rounded-xl">
                        <h4 className="text-lg font-semibold text-white mb-4">Raga Details</h4>
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <p className="text-sm text-gray-400">Aroha (Ascending)</p>
                            <p className="text-white">{result.raga_info.aroha.join(' - ')}</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-400">Avaroha (Descending)</p>
                            <p className="text-white">{result.raga_info.avaroha.join(' - ')}</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-400">Vadi (Main Note)</p>
                            <p className="text-white">{result.raga_info.vadi}</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-400">Samvadi (Second Main Note)</p>
                            <p className="text-white">{result.raga_info.samvadi}</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-400">Time</p>
                            <p className="text-white">{result.raga_info.time}</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-400">Mood</p>
                            <p className="text-white">{result.raga_info.mood}</p>
                          </div>
                        </div>
                      </div>

                      <div className="bg-white/5 p-6 rounded-xl">
                        <h4 className="text-lg font-semibold text-white mb-4">Confidence Scores</h4>
                        <div className="space-y-3">
                          {Object.entries(result.confidence).map(([raga, score]) => (
                            <div key={raga} className="space-y-1">
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-300">{raga}</span>
                                <span className="text-gray-400">{(score * 100).toFixed(1)}%</span>
                              </div>
                              <div className="w-full bg-gray-700 rounded-full h-2">
                                <div
                                  className="bg-yellow-500 h-2 rounded-full"
                                  style={{ width: `${score * 100}%` }}
                                ></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Audio Analysis */}
                      <div className="bg-white/5 p-6 rounded-xl">
                        <h4 className="text-lg font-semibold text-white mb-4">Audio Feature Analysis</h4>
                        {audioAnalysisLoading ? (
                          <div className="flex flex-col items-center justify-center py-12">
                            <Loader2 className="animate-spin text-yellow-400 mb-4" size={48} />
                            <p className="text-white font-medium">Analyzing audio features...</p>
                            <p className="text-gray-400 text-sm mt-2">This may take a few moments</p>
                          </div>
                        ) : audioAnalysisError ? (
                          <div className="bg-red-500/20 border border-red-500/50 text-red-100 p-4 rounded-lg mb-6">
                            <p className="font-medium">Error</p>
                            <p className="text-sm">{audioAnalysisError}</p>
                          </div>
                        ) : audioAnalysis ? (
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                              <p className="text-sm text-gray-400">Mean</p>
                              <p className="text-white">{audioAnalysis.mean}</p>
                            </div>
                            <div>
                              <p className="text-sm text-gray-400">Standard Deviation</p>
                              <p className="text-white">{audioAnalysis.std}</p>
                            </div>
                            <div>
                              <p className="text-sm text-gray-400">Shape</p>
                              <p className="text-white">{audioAnalysis.shape}</p>
                            </div>
                          </div>
                        ) : (
                          <p className="text-gray-400 text-center">No audio analysis available.</p>
                        )}
                      </div>

                      {/* Raga Analysis */}
                      <div className="bg-white/5 p-6 rounded-xl">
                        <h4 className="text-lg font-semibold text-white mb-4">Raga Analysis</h4>
                        {ragaAnalysisLoading ? (
                          <div className="flex flex-col items-center justify-center py-12">
                            <Loader2 className="animate-spin text-yellow-400 mb-4" size={48} />
                            <p className="text-white font-medium">Analyzing raga structure...</p>
                            <p className="text-gray-400 text-sm mt-2">This may take a few moments</p>
                          </div>
                        ) : ragaAnalysisError ? (
                          <div className="bg-red-500/20 border border-red-500/50 text-red-100 p-4 rounded-lg mb-6">
                            <p className="font-medium">Error</p>
                            <p className="text-sm">{ragaAnalysisError}</p>
                          </div>
                        ) : ragaAnalysis ? (
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                              <p className="text-sm text-gray-400">Note Patterns</p>
                              <p className="text-white">{ragaAnalysis.note_patterns}</p>
                            </div>
                            <div>
                              <p className="text-sm text-gray-400">Structure</p>
                              <p className="text-white">{ragaAnalysis.structure}</p>
                            </div>
                            <div>
                              <p className="text-sm text-gray-400">Info</p>
                              <p className="text-white">{ragaAnalysis.info}</p>
                            </div>
                          </div>
                        ) : (
                          <p className="text-gray-400 text-center">No raga analysis available.</p>
                        )}
                      </div>

                      {/* Raga Comparison */}
                      <div className="bg-white/5 p-6 rounded-xl">
                        <h4 className="text-lg font-semibold text-white mb-4">Compare Raga</h4>
                        <div className="flex flex-col sm:flex-row gap-4 mb-4">
                          <select
                            className="flex-1 p-3 rounded-lg bg-gray-800 text-white border border-gray-700 focus:outline-none focus:ring-2 focus:ring-yellow-500"
                            value={selectedCompareId}
                            onChange={(e) => setSelectedCompareId(e.target.value)}
                          >
                            <option value="">Select a raga to compare</option>
                            {compareRagas.map((raga) => (
                              <option key={raga.id} value={raga.id}>
                                {raga.name}
                              </option>
                            ))}
                          </select>
                          <button
                            onClick={handleCompare}
                            disabled={!selectedCompareId || compareLoading}
                            className="px-6 py-3 rounded-full font-medium transition-colors bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600 text-white"
                          >
                            {compareLoading ? (
                              <>
                                <Loader2 className="animate-spin mr-2" size={20} />
                                Comparing...
                              </>
                            ) : (
                              <>
                                <Volume2 className="mr-2" size={20} />
                                Compare
                              </>
                            )}
                          </button>
                        </div>
                        {compareLoading ? (
                          <div className="flex flex-col items-center justify-center py-12">
                            <Loader2 className="animate-spin text-yellow-400 mb-4" size={48} />
                            <p className="text-white font-medium">Comparing ragas...</p>
                            <p className="text-gray-400 text-sm mt-2">This may take a few moments</p>
                          </div>
                        ) : compareError ? (
                          <div className="bg-red-500/20 border border-red-500/50 text-red-100 p-4 rounded-lg mb-6">
                            <p className="font-medium">Error</p>
                            <p className="text-sm">{compareError}</p>
                          </div>
                        ) : compareResult ? (
                          <div className="space-y-6">
                            <div className="bg-gradient-to-r from-green-500/20 to-blue-500/20 p-6 rounded-xl border border-green-500/30">
                              <h5 className="text-xl font-bold text-white mb-4">Comparison Results</h5>
                              <p className="text-white">{compareResult.message}</p>
                            </div>
                            {compareResult.features && (
                              <div className="bg-white/5 p-6 rounded-xl">
                                <h5 className="text-lg font-semibold text-white mb-4">Feature Differences</h5>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                  {Object.entries(compareResult.features).map(([feature, value]) => (
                                    <div key={feature} className="space-y-1">
                                      <div className="flex justify-between text-sm">
                                        <span className="text-gray-300">{feature}</span>
                                        <span className="text-gray-400">{value}</span>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        ) : (
                          <p className="text-gray-400 text-center">No comparison results available.</p>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                      <div className="bg-white/10 p-4 rounded-full mb-4">
                        <Mic size={32} className="text-gray-400" />
                      </div>
                      <h3 className="text-xl font-medium text-white mb-2">No Analysis Done Yet</h3>
                      <p className="text-gray-400 max-w-md">
                        Upload an audio file and click "Detect Raga" to analyze the raga
                      </p>
                    </div>
                  )}
                </div>
              </React.Fragment>
            } />
            <Route path="/ragas" element={<RagaList />} />
          </Routes>
        </div>

        {/* Footer */}
        <div className="text-center text-gray-500 text-sm mt-12">
          <p>© {new Date().getFullYear()} Raga Detection System | Made with ❤️ for Indian Classical Music</p>
        </div>
      </div>
    </Router>
  );
}

export default App;
