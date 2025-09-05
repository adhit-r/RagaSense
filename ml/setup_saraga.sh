#!/bin/bash

# Saraga Integration Setup Script for RagaSense
# Setting up MTG's professional raga dataset integration

echo "🎵 Setting up Saraga integration for RagaSense..."

# Activate virtual environment
source ../raga_env/bin/activate

# Install additional dependencies
echo "📦 Installing Saraga dependencies..."
pip install pandas requests

# Run Saraga integration
echo "🔧 Running Saraga integration..."
python saraga_integration.py

# Check results
echo "📊 Checking integration results..."
if [ -d "saraga_data" ]; then
    echo "✅ Saraga integration data created"
    ls -la saraga_data/
else
    echo "❌ Saraga integration failed"
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Register at https://dunya.compmusic.upf.edu/ for API access"
echo "2. Get API token and update saraga_integration.py"
echo "3. Run: python saraga_integration.py with API token"
echo "4. Download actual Saraga data"
echo "5. Integrate with YuE classifier"
echo ""
echo "📚 Resources:"
echo "- Saraga Repository: https://github.com/MTG/saraga"
echo "- Dunya API: https://dunya.commpusic.upf.edu/"
echo "- Integration Plan: docs/SARAGA_INTEGRATION_PLAN.md"
echo ""
echo "🎉 Saraga integration setup complete!"
