import { Database, Search, Filter } from 'lucide-react';

export function RagaList() {
  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold">Browse Ragas</h1>
        <p className="text-xl text-muted-foreground">
          Explore thousands of ragas from Hindustani and Carnatic traditions
        </p>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search ragas..."
            className="w-full pl-10 pr-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>
        <div className="flex gap-2">
          <select className="px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary">
            <option value="">All Traditions</option>
            <option value="hindustani">Hindustani</option>
            <option value="carnatic">Carnatic</option>
          </select>
          <select className="px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary">
            <option value="">All Times</option>
            <option value="morning">Morning</option>
            <option value="afternoon">Afternoon</option>
            <option value="evening">Evening</option>
            <option value="night">Night</option>
          </select>
        </div>
      </div>

      {/* Raga Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Sample Raga Cards */}
        {[
          { name: 'Yaman', tradition: 'Hindustani', time: 'Evening', description: 'A beautiful evening raga with romantic and devotional qualities.' },
          { name: 'Bhairav', tradition: 'Hindustani', time: 'Morning', description: 'A morning raga with deep, meditative characteristics.' },
          { name: 'Malkauns', tradition: 'Hindustani', time: 'Night', description: 'A night raga with mystical and introspective qualities.' },
          { name: 'Bilawal', tradition: 'Hindustani', time: 'Morning', description: 'A morning raga with bright and cheerful characteristics.' },
          { name: 'Khamaj', tradition: 'Hindustani', time: 'Evening', description: 'An evening raga with romantic and light qualities.' },
          { name: 'Bhairavi', tradition: 'Hindustani', time: 'Morning', description: 'A morning raga with devotional and serious qualities.' },
        ].map((raga, index) => (
          <div key={index} className="border rounded-lg p-6 hover:shadow-lg transition-shadow">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-semibold">{raga.name}</h3>
              <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                {raga.tradition}
              </span>
            </div>
            <p className="text-sm text-gray-600 mb-3">{raga.time}</p>
            <p className="text-sm text-muted-foreground">{raga.description}</p>
            <button className="mt-4 text-primary hover:text-primary/80 text-sm font-medium">
              Learn More →
            </button>
          </div>
        ))}
      </div>

      {/* Load More */}
      <div className="text-center">
        <button className="bg-primary text-primary-foreground px-8 py-3 rounded-lg font-medium hover:bg-primary/90 transition-colors">
          Load More Ragas
        </button>
      </div>
    </div>
  );
}
