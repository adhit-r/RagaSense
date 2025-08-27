---
name: Raga Detection Issue
about: Report issues with raga detection accuracy or performance
title: '[RAGA-DETECTION] '
labels: ['raga-detection', 'ml', 'needs-triage']
assignees: ''
---

## Detection Issue Description
Describe the raga detection problem you encountered.

## Audio Sample Information
- **Audio Format**: [e.g. WAV, MP3, OGG]
- **Duration**: [e.g. 30 seconds]
- **Quality**: [e.g. High, Medium, Low]
- **Source**: [e.g. Live recording, Uploaded file, Generated]

## Expected Raga
What raga did you expect to be detected?

## Actual Detection Result
What raga was actually detected?

```
{
  "predicted_raga": "Actual Raga",
  "confidence": 0.85,
  "top_predictions": [
    {"raga": "Actual Raga", "probability": 0.85, "confidence": "High"},
    {"raga": "Other Raga", "probability": 0.10, "confidence": "Low"}
  ]
}
```

## Audio Sample Details
- **Artist**: [if known]
- **Composition**: [if known]
- **Time of Day**: [if relevant - morning, evening, night]
- **Season**: [if relevant - monsoon, summer, etc.]

## Technical Details
- **Model Version**: [e.g. v1.0.0]
- **Processing Time**: [e.g. 0.06 seconds]
- **File Size**: [e.g. 2.5 MB]

## Steps to Reproduce
1. Upload/record audio sample
2. Wait for processing
3. View detection result
4. Compare with expected result

## Additional Context
- Is this a known raga?
- Are there specific characteristics that might affect detection?
- Any background noise or quality issues?

## Checklist
- [ ] I have provided the expected vs actual raga
- [ ] I have included audio sample details
- [ ] I can reproduce this issue with the same audio
- [ ] I have checked if this is a known limitation
