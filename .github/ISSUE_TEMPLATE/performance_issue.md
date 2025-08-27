---
name: Performance Issue
about: Report performance problems, slow loading, or optimization needs
title: '[PERFORMANCE] '
labels: ['performance', 'optimization', 'speed', 'needs-triage']
assignees: ''
---

## Issue Description
Describe the performance problem you encountered.

## Issue Type
- [ ] Slow Loading Time
- [ ] High Memory Usage
- [ ] CPU Performance Issue
- [ ] Network Latency
- [ ] Database Performance
- [ ] Frontend Rendering
- [ ] ML Model Inference
- [ ] Build Time Issue
- [ ] Other

## Performance Metrics
- **Loading Time**: [e.g. 5+ seconds]
- **Memory Usage**: [e.g. 500MB+]
- **CPU Usage**: [e.g. 80%+]
- **Network Requests**: [e.g. 50+ requests]
- **Bundle Size**: [e.g. 2MB+]

## Affected Component
- [ ] Frontend (Lynx)
- [ ] Backend (FastAPI)
- [ ] Database (Convex)
- [ ] ML Model
- [ ] Build Process
- [ ] All Components

## Environment
- **Platform**: [Web, iOS, Android, All]
- **Device**: [iPhone 14, Samsung Galaxy, Desktop, etc.]
- **Network**: [WiFi, 4G, 3G, Slow connection]
- **Browser**: [Chrome, Safari, Firefox] (if web)

## Steps to Reproduce
1. Open app on '...'
2. Navigate to '...'
3. Perform action '...'
4. Notice slow performance

## Expected Performance
What should the performance be like?

## Actual Performance
What is the current performance like?

## Performance Data
If you have specific metrics, include them:

```json
{
  "loading_time": "5.2 seconds",
  "memory_usage": "450MB",
  "cpu_usage": "75%",
  "network_requests": "45"
}
```

## Browser Performance Tools
If applicable, include data from:
- **Lighthouse**: Performance score
- **Chrome DevTools**: Network tab, Performance tab
- **WebPageTest**: Detailed metrics

## Additional Context
- Does this happen on all devices?
- Is this a regression from a previous version?
- Any recent changes that might have caused this?
- Does this happen consistently or intermittently?

## Checklist
- [ ] I have tested on multiple devices/platforms
- [ ] I have cleared cache and tried again
- [ ] I have checked network conditions
- [ ] I have provided specific performance metrics
- [ ] This is not a duplicate of an existing issue
