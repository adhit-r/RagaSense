import React, { useEffect, useState } from 'react';

const RagaList = () => {
  const [ragas, setRagas] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchRagas = async () => {
      try {
        const res = await fetch('/api/ragas');
        if (!res.ok) throw new Error('Failed to fetch ragas');
        const data = await res.json();
        setRagas(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchRagas();
  }, []);

  if (loading) return <div className="p-8 text-center">Loading ragas...</div>;
  if (error) return <div className="p-8 text-center text-red-500">Error: {error}</div>;

  return (
    <div className="max-w-2xl mx-auto p-8">
      <h1 className="text-2xl font-bold mb-4">Raga List</h1>
      <ul className="divide-y divide-gray-200 bg-white rounded shadow">
        {ragas.map((raga, idx) => (
          <li key={idx} className="p-4 hover:bg-gray-50">{raga}</li>
        ))}
      </ul>
    </div>
  );
};

export default RagaList; 