---
name: Convex Integration Issue
about: Report issues with Convex database, authentication, or real-time features
title: '[CONVEX] '
labels: ['convex', 'database', 'authentication', 'needs-triage']
assignees: ''
---

## Issue Description
Describe the Convex-related problem you encountered.

## Issue Type
- [ ] Database Schema Issue
- [ ] Authentication Problem
- [ ] Real-time Sync Issue
- [ ] Query/Mutation Error
- [ ] Performance Issue
- [ ] Deployment Problem
- [ ] Other

## Error Details
If applicable, provide the error message or stack trace:

```javascript
// Example error
Error: ConvexError: Invalid argument
  at api.users.getCurrentUser (convex/users.ts:15:3)
```

## Steps to Reproduce
1. Navigate to '...'
2. Perform action '...'
3. Expected: '...'
4. Actual: '...'

## Environment
- **Platform**: [Web, iOS, Android]
- **Browser**: [if applicable]
- **Convex Environment**: [dev, prod]
- **User Authentication**: [Logged in, Guest, Specific provider]

## Database Context
- **Table**: [e.g. users, ragaDetections, files]
- **Operation**: [Query, Mutation, Action]
- **Data Size**: [e.g. Number of records affected]

## Authentication Context
- **Auth Provider**: [Convex Auth, Google, GitHub, etc.]
- **User State**: [Authenticated, Unauthenticated, Token expired]
- **Permissions**: [Admin, User, Guest]

## Expected Behavior
A clear description of what should happen.

## Actual Behavior
A clear description of what actually happened.

## Screenshots/Logs
If applicable, add screenshots or console logs.

## Additional Context
- Is this happening for all users or specific users?
- Does this happen consistently or intermittently?
- Any recent changes to Convex functions or schema?

## Checklist
- [ ] I have checked the Convex dashboard for errors
- [ ] I have verified my authentication status
- [ ] I have tested with different data
- [ ] I have checked the browser console for errors
- [ ] This is not a duplicate of an existing issue
