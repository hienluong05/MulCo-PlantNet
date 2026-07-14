import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, X, Leaf, Sparkles, AlertCircle, RefreshCw, Layers } from 'lucide-react';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0];
      handleFileSelection(selectedFile);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleFileSelection = (selectedFile) => {
    setFile(selectedFile);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(selectedFile);
    setResult(null);
    setError(null);
  };

  const clearSelection = () => {
    setFile(null);
    setPreview('');
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    if (caption.trim()) {
      formData.append('caption', caption);
    }

    try {
      const response = await axios.post('/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError('An error occurred during prediction. Please make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>MulCo-PlantNet</h1>
        <p>Multimodal Plant Disease Identification with Grad-CAM Visualization</p>
      </div>

      <div className="main-content">
        <div className="upload-section">
          {!preview ? (
            <div 
              className="dropzone"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={48} className="dropzone-icon" />
              <h3>Drag & Drop an image here</h3>
              <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                or click to browse from your computer
              </p>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileInput} 
                accept="image/*" 
                style={{ display: 'none' }} 
              />
            </div>
          ) : (
            <div className="preview-container">
              <img src={preview} alt="Preview" className="preview-image" />
              <button className="remove-btn" onClick={clearSelection}>
                <X size={16} />
              </button>
            </div>
          )}

          <div className="input-group">
            <label htmlFor="caption">Pathological Description (Optional)</label>
            <textarea
              id="caption"
              className="textarea"
              placeholder="Leave empty to auto-generate CoT description using Gemini API, or type specific symptoms (e.g., 'Yellowing leaves with concentric brown spots...')"
              value={caption}
              onChange={(e) => setCaption(e.target.value)}
            />
          </div>

          <button 
            className="submit-btn" 
            onClick={handleSubmit} 
            disabled={!file || loading}
          >
            {loading ? (
              <><RefreshCw className="loader" size={20} /> Processing...</>
            ) : (
              <><Leaf size={20} /> Analyze Leaf</>
            )}
          </button>

          {error && (
            <div style={{ color: 'var(--error)', display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '1rem' }}>
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}
        </div>

        <div className="result-section">
          {result ? (
            <div className="result-card">
              <div className="result-header">
                <h3 className="result-title">Prediction Result</h3>
                <div className="prediction-name">
                  {result.prediction.replace(/_/g, ' ')}
                </div>
              </div>

              <div className="confidence-section">
                <div className="confidence-bar-container">
                  <div 
                    className="confidence-bar" 
                    style={{ width: `${Math.round(result.confidence * 100)}%` }}
                  ></div>
                </div>
                <div className="confidence-text">
                  <span>Confidence Score</span>
                  <strong>{Math.round(result.confidence * 100)}%</strong>
                </div>
              </div>

              {result.gradcam_base64 && (
                <div className="cam-image-container">
                  <h4 className="cam-title"><Layers size={18}/> Grad-CAM Heatmap</h4>
                  <img 
                    src={result.gradcam_base64} 
                    alt="Grad-CAM" 
                    style={{ width: '100%', borderRadius: '12px', marginTop: '0.5rem', border: '1px solid var(--border-color)' }}
                  />
                </div>
              )}

              <div className="caption-card">
                <div className="caption-title">
                  <Sparkles size={16} /> 
                  {result.generated_caption ? 'Auto-Generated Caption (Gemini)' : 'User Provided Caption'}
                </div>
                <div className="caption-text">
                  {result.caption}
                </div>
              </div>
            </div>
          ) : (
            <div className="result-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '300px', opacity: 0.5 }}>
              <p>Upload an image and analyze to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
