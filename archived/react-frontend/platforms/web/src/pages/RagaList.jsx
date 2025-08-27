import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Search, Filter, Music, Clock, ArrowRight } from 'lucide-react';
import '../styles/RagaList.css';

// Mock data - replace with API call in production
const mockRagas = [
  {
    id: 1,
    name: 'Mohanam',
    arohanam: 'S R2 G3 P D2 S',
    avarohanam: 'S D2 P G3 R2 S',
    time: 'Evening',
    mood: 'Joyful',
    description: 'Mohanam is a popular raga in Carnatic music known for its sweet and joyful nature.',
    similar: ['Shuddha Saveri', 'Bhoopali']
  },
  {
    id: 2,
    name: 'Kalyani',
    arohanam: 'S R2 G3 M2 P D2 N3 S',
    avarohanam: 'S N3 D2 P M2 G3 R2 S',
    time: 'Night',
    mood: 'Majestic',
    description: 'Kalyani is a major raga that is majestic and considered to be full of grace.',
    similar: ['Shankarabharanam', 'Khamas']
  },
  // Add more ragas as needed
];

export function RagaList() {
  const [searchTerm, setSearchTerm] = useState('');
  const [timeFilter, setTimeFilter] = useState('all');
  const [filteredRagas, setFilteredRagas] = useState(mockRagas);

  useEffect(() => {
    const results = mockRagas.filter(raga => {
      const matchesSearch = raga.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         raga.arohanam.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesTime = timeFilter === 'all' || raga.time.toLowerCase() === timeFilter.toLowerCase();
      return matchesSearch && matchesTime;
    });
    setFilteredRagas(results);
  }, [searchTerm, timeFilter]);

  return (
    <div className="raga-list-container">
      <h1>Explore Ragas</h1>
      
      <div className="filters">
        <div className="search-bar">
          <Search className="search-icon" />
          <input
            type="text"
            placeholder="Search ragas by name or notes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        
        <div className="time-filter">
          <Filter className="filter-icon" />
          <select
            value={timeFilter}
            onChange={(e) => setTimeFilter(e.target.value)}
            aria-label="Filter by time of day"
          >
            <option value="all">All Times</option>
            <option value="morning">Morning</option>
            <option value="afternoon">Afternoon</option>
            <option value="evening">Evening</option>
            <option value="night">Night</option>
          </select>
        </div>
      </div>
      
      <div className="raga-grid">
        {filteredRagas.length > 0 ? (
          filteredRagas.map((raga) => (
            <div key={raga.id} className="raga-card">
              <div className="raga-header">
                <Music className="raga-icon" />
                <h2>{raga.name}</h2>
                <span className={`time-badge ${raga.time.toLowerCase()}`}>
                  <Clock size={14} />
                  {raga.time}
                </span>
              </div>
              
              <div className="raga-details">
                <div className="detail-row">
                  <span className="detail-label">Arohanam:</span>
                  <span className="detail-value notes">{raga.arohanam}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Avarohanam:</span>
                  <span className="detail-value notes">{raga.avarohanam}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Mood:</span>
                  <span className="detail-value">{raga.mood}</span>
                </div>
                
                {raga.similar && raga.similar.length > 0 && (
                  <div className="similar-ragas">
                    <span className="detail-label">Similar to:</span>
                    <div className="similar-tags">
                      {raga.similar.map((similarRaga, idx) => (
                        <span key={idx} className="similar-tag">{similarRaga}</span>
                      ))}
                    </div>
                  </div>
                )}
                
                <p className="raga-description">{raga.description}</p>
              </div>
              
              <Link to={`/ragas/${raga.id}`} className="view-details">
                View Details <ArrowRight size={16} />
              </Link>
            </div>
          ))
        ) : (
          <div className="no-results">
            <h3>No ragas found</h3>
            <p>Try adjusting your search or filters</p>
          </div>
        )}
      </div>
    </div>
  );
}
