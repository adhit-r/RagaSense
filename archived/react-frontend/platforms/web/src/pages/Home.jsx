import React from 'react';
import { Link } from 'react-router-dom';
import { Mic, Music, Info, Clock, Headphones, Upload } from 'lucide-react';
import '../styles/Home.css';

export function Home() {
  return (
    <div className="home">
      <section className="hero">
        <div className="hero-content">
          <h1>Discover the Magic of Indian Classical Music</h1>
          <p className="subtitle">
            Identify ragas in real-time with our advanced audio analysis technology.
            Perfect for music students, performers, and enthusiasts.
          </p>
          
          <div className="cta-buttons">
            <Link to="/detect" className="btn primary">
              <Mic className="btn-icon" />
              <span>Detect Raga</span>
            </Link>
            <Link to="/ragas" className="btn secondary">
              <Music className="btn-icon" />
              <span>Explore Ragas</span>
            </Link>
          </div>
        </div>
        
        <div className="hero-image">
          <div className="visualizer-placeholder">
            <div className="visualizer-animation">
              {[...Array(8)].map((_, i) => (
                <div key={i} className="bar" style={{
                  animationDelay: `${i * 0.1}s`,
                  height: `${40 + Math.random() * 60}%`
                }} />
              ))}
            </div>
          </div>
        </div>
      </section>
      
      <section className="features">
        <h2>Why Choose Raga Detect?</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">
              <Mic />
            </div>
            <h3>Real-time Analysis</h3>
            <p>Instantly identify ragas from live audio or recordings with our advanced signal processing.</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">
              <Info />
            </div>
            <h3>Detailed Information</h3>
            <p>Get comprehensive details about each raga, including arohanam, avarohanam, and more.</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">
              <Clock />
            </div>
            <h3>Time-based Suggestions</h3>
            <p>Discover which ragas are traditionally performed at different times of the day.</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">
              <Headphones />
            </div>
            <h3>Audio Examples</h3>
            <p>Listen to high-quality samples of each raga to train your ear.</p>
          </div>
        </div>
      </section>
      
      <section className="how-it-works">
        <h2>How It Works</h2>
        <div className="steps">
          <div className="step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h3>Record or Upload</h3>
              <p>Use your microphone to record live music or upload an existing audio file.</p>
            </div>
          </div>
          
          <div className="step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h3>Analyze</h3>
              <p>Our advanced algorithms analyze the audio to identify the musical notes and patterns.</p>
            </div>
          </div>
          
          <div className="step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h3>Get Results</h3>
              <p>Receive instant feedback on the detected raga along with detailed information.</p>
            </div>
          </div>
        </div>
        
        <div className="cta-section">
          <Link to="/detect" className="btn primary large">
            <Upload className="btn-icon" />
            <span>Try It Now</span>
          </Link>
        </div>
      </section>
      
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-logo">
            <Mic className="logo-icon" />
            <span>Raga Detect</span>
          </div>
          <p className="copyright">
            &copy; {new Date().getFullYear()} Raga Detect. All rights reserved.
          </p>
          <div className="footer-links">
            <a href="#privacy">Privacy Policy</a>
            <span className="divider">•</span>
            <a href="#terms">Terms of Service</a>
            <span className="divider">•</span>
            <a href="#contact">Contact Us</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
