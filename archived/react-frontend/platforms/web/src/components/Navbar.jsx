import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { Moon, Sun, Mic, Music, Home } from 'lucide-react';
import '../styles/Navbar.css';

export function Navbar() {
  const { theme, toggleTheme } = useTheme();
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path ? 'active' : '';
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-brand">
          <Mic className="logo-icon" />
          <span>Raga Detect</span>
        </Link>
        
        <div className="nav-links">
          <Link to="/" className={`nav-link ${isActive('/')}`}>
            <Home className="nav-icon" />
            <span>Home</span>
          </Link>
          <Link to="/detect" className={`nav-link ${isActive('/detect')}`}>
            <Mic className="nav-icon" />
            <span>Detect</span>
          </Link>
          <Link to="/ragas" className={`nav-link ${isActive('/ragas')}`}>
            <Music className="nav-icon" />
            <span>Ragas</span>
          </Link>
        </div>
        
        <button 
          className="theme-toggle" 
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
        >
          {theme === 'light' ? (
            <Moon className="theme-icon" />
          ) : (
            <Sun className="theme-icon" />
          )}
        </button>
      </div>
    </nav>
  );
}
