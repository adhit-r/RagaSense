import React from 'react';
import PropTypes from 'prop-types';
import { Music, Clock, Zap, Activity } from 'lucide-react';
import '../styles/RagaResult.css';

export function RagaResult({ result }) {
  if (!result) return null;

  const { 
    raga, 
    confidence, 
    matchingNotes = [],
    mood,
    time,
    similarRagas = []
  } = result;

  // Calculate confidence percentage and color
  const confidencePercent = Math.round(confidence * 100);
  const confidenceColor = 
    confidence > 0.8 ? '#4CAF50' : 
    confidence > 0.6 ? '#FFC107' : '#F44336';

  // Format the confidence display
  const formatConfidence = (value) => {
    return `${Math.round(value * 100)}%`;
  };
  
  // Get mood emoji based on mood text
  const getMoodEmoji = (mood) => {
    const moodMap = {
      'joyful': '😊',
      'peaceful': '😌',
      'devotional': '🙏',
      'romantic': '💕',
      'energetic': '⚡',
      'melancholic': '😔',
      'mystical': '🔮',
      'heroic': '🦸',
    };
    return moodMap[mood.toLowerCase()] || '🎵';
  };

  return (
    <div className="raga-result">
      <h2>Detected Raga</h2>
      
      <div className="result-card">
        <div className="raga-name">
          <Music size={32} className="icon" />
          {raga}
        </div>
        
        <div className="confidence-meter">
          <div className="confidence-label">Confidence:</div>
          <div className="confidence-bar">
            <div 
              className="confidence-fill"
              style={{
                width: `${confidencePercent}%`,
                backgroundColor: confidenceColor
              }}
            ></div>
            <div className="confidence-value">{formatConfidence(confidence)}</div>
          </div>
        </div>
        
        <div className="raga-details">
          {mood && (
            <div className="detail-item">
              <span className="detail-label">
                <Zap size={16} className="icon" /> Mood:
              </span>
              <span className="detail-value">
                {getMoodEmoji(mood)} {mood}
              </span>
            </div>
          )}
          
          {time && (
            <div className="detail-item">
              <span className="detail-label">
                <Clock size={16} className="icon" /> Best Time:
              </span>
              <span className="detail-value">{time}</span>
            </div>
          )}
          
          {matchingNotes.length > 0 && (
            <div className="detail-item notes">
              <span className="detail-label">
                <Activity size={16} className="icon" /> Notes:
              </span>
              <div className="notes-container">
                {matchingNotes.map((note, index) => {
                  // Split compound notes (e.g., 'S R2 G3')
                  const notes = note.split(' ');
                  return (
                    <React.Fragment key={index}>
                      {notes.map((n, i) => (
                        <span key={`${index}-${i}`} className="note">
                          {n}
                        </span>
                      ))}
                      {index < matchingNotes.length - 1 && ' - '}
                    </React.Fragment>
                  );
                })}
              </div>
            </div>
          )}
        </div>
        
        {similarRagas.length > 0 && (
          <div className="similar-ragas">
            <h4>Similar Ragas:</h4>
            <div className="similar-list">
              {similarRagas.slice(0, 3).map((raga, index) => (
                <div key={index} className="similar-raga">
                  <span className="similar-name">{raga.name}</span>
                  <div className="similar-confidence">
                    <div 
                      className="similar-bar"
                      style={{
                        width: `${raga.confidence * 100}%`,
                        backgroundColor: raga.confidence > 0.7 ? '#4CAF50' : '#FFC107'
                      }}
                    ></div>
                    <span className="similar-percent">
                      {formatConfidence(raga.confidence)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      <div className="action-buttons">
        <button className="btn primary">Learn More</button>
        <button className="btn secondary">Save Result</button>
      </div>
    </div>
  );
}

RagaResult.propTypes = {
  result: PropTypes.shape({
    raga: PropTypes.string.isRequired,
    confidence: PropTypes.number.isRequired,
    matchingNotes: PropTypes.arrayOf(PropTypes.string),
    mood: PropTypes.string,
    time: PropTypes.string,
    similarRagas: PropTypes.arrayOf(
      PropTypes.shape({
        name: PropTypes.string.isRequired,
        confidence: PropTypes.number.isRequired
      })
    )
  })
};

export { RagaResult };
