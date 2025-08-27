import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Search, ArrowUpDown } from 'lucide-react';

interface Raga {
  name: string;
  type: string;
  time: string;
  mood: string;
  aroha: string[];
  avaroha: string[];
}

export function RagaList() {
  const [searchTerm, setSearchTerm] = useState('');
  const [ragas, setRagas] = useState<Raga[]>([]);
  const [filteredRagas, setFilteredRagas] = useState<Raga[]>([]);

  // Mock data for demonstration
  useEffect(() => {
    const mockRagas: Raga[] = [
      {
        name: 'Yaman',
        type: 'Hindustani',
        time: 'Evening',
        mood: 'Peaceful, Romantic',
        aroha: ['Ni', 'Re', 'Ga', 'Ma#', 'Pa', 'Dha', 'Ni', 'Sa'],
        avaroha: ['Sa', 'Ni', 'Dha', 'Pa', 'Ma#', 'Ga', 'Re', 'Sa']
      },
      // Add more ragas as needed
    ];
    
    setRagas(mockRagas);
    setFilteredRagas(mockRagas);
  }, []);

  // Filter ragas based on search term
  useEffect(() => {
    if (!searchTerm.trim()) {
      setFilteredRagas(ragas);
      return;
    }

    const filtered = ragas.filter(raga =>
      raga.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      raga.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
      raga.mood.toLowerCase().includes(searchTerm.toLowerCase())
    );
    
    setFilteredRagas(filtered);
  }, [searchTerm, ragas]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <h1 className="text-3xl font-bold tracking-tight">Raga Library</h1>
        <div className="relative w-full md:w-96">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search ragas..."
            className="pl-10"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {filteredRagas.map((raga, index) => (
          <div key={index} className="border rounded-lg overflow-hidden">
            <div className="p-6">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-xl font-semibold">{raga.name}</h3>
                  <p className="text-sm text-muted-foreground">{raga.type}</p>
                </div>
                <span className="px-2 py-1 text-xs rounded-full bg-primary/10 text-primary">
                  {raga.time}
                </span>
              </div>
              
              <div className="mt-4 space-y-2">
                <div>
                  <p className="text-sm font-medium">Aroha</p>
                  <p className="text-sm">{raga.aroha.join(' ')}</p>
                </div>
                <div>
                  <p className="text-sm font-medium">Avaroha</p>
                  <p className="text-sm">{raga.avaroha.join(' ')}</p>
                </div>
                <div>
                  <p className="text-sm font-medium">Mood</p>
                  <p className="text-sm">{raga.mood}</p>
                </div>
              </div>
              
              <Button variant="outline" size="sm" className="mt-4 w-full">
                Learn More
              </Button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
