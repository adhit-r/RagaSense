#!/usr/bin/env python3
"""
RagaSense Roadmap Issue Creator

This script creates GitHub issues from the roadmap tasks defined in docs/ROADMAP.md.
It uses the GitHub API to create issues with proper labels, milestones, and descriptions.

Usage:
    python scripts/create_roadmap_issues.py

Requirements:
    - GitHub Personal Access Token with repo scope
    - requests library: pip install requests
"""

import os
import re
import requests
import json
from typing import Dict, List, Optional

# Configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_OWNER = 'adhit-r'
REPO_NAME = 'RagaSense'
ROADMAP_FILE = 'docs/ROADMAP.md'

# GitHub API configuration
GITHUB_API_BASE = 'https://api.github.com'
HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json',
    'Content-Type': 'application/json'
}

class RoadmapIssueCreator:
    def __init__(self):
        self.issues_created = []
        self.milestones = {}
        self.labels = {}
        
    def create_milestones(self):
        """Create milestones for each development phase"""
        milestones_data = [
            {
                'title': 'Phase 1: Foundation & Core Features',
                'description': 'Complete core ML model, frontend polish, and database integration',
                'due_on': '2024-03-31T23:59:59Z'
            },
            {
                'title': 'Phase 2: Advanced Features',
                'description': 'Implement AI music generation, social features, and advanced analytics',
                'due_on': '2024-06-30T23:59:59Z'
            },
            {
                'title': 'Phase 3: Enterprise & Scale',
                'description': 'Add enterprise features, performance optimization, and scalability',
                'due_on': '2024-09-30T23:59:59Z'
            },
            {
                'title': 'Phase 4: Innovation & Research',
                'description': 'Advanced AI features, research collaboration, and educational platform',
                'due_on': '2024-12-31T23:59:59Z'
            }
        ]
        
        for milestone_data in milestones_data:
            response = requests.post(
                f'{GITHUB_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/milestones',
                headers=HEADERS,
                json=milestone_data
            )
            
            if response.status_code == 201:
                milestone = response.json()
                self.milestones[milestone_data['title']] = milestone['number']
                print(f"✅ Created milestone: {milestone_data['title']}")
            else:
                print(f"❌ Failed to create milestone: {milestone_data['title']}")
                print(f"Error: {response.text}")
    
    def create_labels(self):
        """Create labels for issue categorization"""
        labels_data = [
            {'name': 'phase-1', 'color': '0e8a16', 'description': 'Phase 1: Foundation & Core Features'},
            {'name': 'phase-2', 'color': 'fbca04', 'description': 'Phase 2: Advanced Features'},
            {'name': 'phase-3', 'color': 'd93f0b', 'description': 'Phase 3: Enterprise & Scale'},
            {'name': 'phase-4', 'color': '5319e7', 'description': 'Phase 4: Innovation & Research'},
            {'name': 'ml-model', 'color': '1d76db', 'description': 'Machine Learning Model'},
            {'name': 'frontend', 'color': 'c2e0c6', 'description': 'Frontend Development'},
            {'name': 'backend', 'color': 'd4c5f9', 'description': 'Backend Development'},
            {'name': 'database', 'color': 'fef2c0', 'description': 'Database & Authentication'},
            {'name': 'ai-generation', 'color': 'bfdadc', 'description': 'AI Music Generation'},
            {'name': 'social-features', 'color': 'f9d0c4', 'description': 'Social & Collaborative Features'},
            {'name': 'analytics', 'color': 'c5def5', 'description': 'Analytics & Monitoring'},
            {'name': 'performance', 'color': 'fef7c0', 'description': 'Performance & Optimization'},
            {'name': 'research', 'color': 'd1ecf1', 'description': 'Research & Education'},
            {'name': 'high-priority', 'color': 'd73a4a', 'description': 'High Priority - Must Have'},
            {'name': 'medium-priority', 'color': 'fbca04', 'description': 'Medium Priority - Should Have'},
            {'name': 'low-priority', 'color': '0e8a16', 'description': 'Low Priority - Nice to Have'},
            {'name': 'enhancement', 'color': 'a2eeef', 'description': 'New feature or request'},
            {'name': 'roadmap', 'color': '5319e7', 'description': 'Part of the development roadmap'}
        ]
        
        for label_data in labels_data:
            response = requests.post(
                f'{GITHUB_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/labels',
                headers=HEADERS,
                json=label_data
            )
            
            if response.status_code in [201, 422]:  # 422 means label already exists
                print(f"✅ Label ready: {label_data['name']}")
            else:
                print(f"❌ Failed to create label: {label_data['name']}")
                print(f"Error: {response.text}")
    
    def parse_roadmap_file(self) -> List[Dict]:
        """Parse the roadmap file and extract issue information"""
        issues = []
        
        if not os.path.exists(ROADMAP_FILE):
            print(f"❌ Roadmap file not found: {ROADMAP_FILE}")
            return issues
        
        with open(ROADMAP_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse issues from the roadmap
        issue_pattern = r'- \[ \] \*\*Issue #(\d+)\*\*: (.+?)(?=\n- \[ \]|\n\n|\Z)'
        matches = re.findall(issue_pattern, content, re.DOTALL)
        
        for issue_num, issue_content in matches:
            # Extract title and description
            lines = issue_content.strip().split('\n')
            title = lines[0].strip()
            description_lines = [line.strip() for line in lines[1:] if line.strip()]
            description = '\n'.join(description_lines)
            
            # Determine phase and labels
            phase = self._determine_phase(issue_num)
            labels = self._determine_labels(title, description, phase)
            priority = self._determine_priority(issue_num)
            
            issues.append({
                'number': int(issue_num),
                'title': title,
                'description': description,
                'phase': phase,
                'labels': labels,
                'priority': priority
            })
        
        return sorted(issues, key=lambda x: x['number'])
    
    def _determine_phase(self, issue_num: str) -> str:
        """Determine which phase an issue belongs to based on its number"""
        num = int(issue_num)
        if num <= 8:
            return 'Phase 1: Foundation & Core Features'
        elif num <= 15:
            return 'Phase 2: Advanced Features'
        elif num <= 19:
            return 'Phase 3: Enterprise & Scale'
        else:
            return 'Phase 4: Innovation & Research'
    
    def _determine_labels(self, title: str, description: str, phase: str) -> List[str]:
        """Determine appropriate labels for an issue"""
        labels = ['roadmap', 'enhancement']
        
        # Phase labels
        if 'Phase 1' in phase:
            labels.append('phase-1')
        elif 'Phase 2' in phase:
            labels.append('phase-2')
        elif 'Phase 3' in phase:
            labels.append('phase-3')
        elif 'Phase 4' in phase:
            labels.append('phase-4')
        
        # Category labels
        title_lower = title.lower()
        desc_lower = description.lower()
        
        if any(word in title_lower or word in desc_lower for word in ['ml', 'model', 'training', 'data', 'accuracy']):
            labels.append('ml-model')
        
        if any(word in title_lower or word in desc_lower for word in ['frontend', 'ui', 'mobile', 'app', 'lynx']):
            labels.append('frontend')
        
        if any(word in title_lower or word in desc_lower for word in ['backend', 'api', 'server', 'fastapi']):
            labels.append('backend')
        
        if any(word in title_lower or word in desc_lower for word in ['database', 'auth', 'convex', 'analytics']):
            labels.append('database')
            if 'analytics' in title_lower or 'analytics' in desc_lower:
                labels.append('analytics')
        
        if any(word in title_lower or word in desc_lower for word in ['generation', 'compose', 'music', 'ai']):
            labels.append('ai-generation')
        
        if any(word in title_lower or word in desc_lower for word in ['social', 'share', 'community', 'collaborative']):
            labels.append('social-features')
        
        if any(word in title_lower or word in desc_lower for word in ['performance', 'optimization', 'scale', 'speed']):
            labels.append('performance')
        
        if any(word in title_lower or word in desc_lower for word in ['research', 'education', 'academic', 'tutorial']):
            labels.append('research')
        
        return labels
    
    def _determine_priority(self, issue_num: str) -> str:
        """Determine priority based on issue number"""
        num = int(issue_num)
        high_priority = [1, 2, 4, 7, 9]
        medium_priority = [3, 5, 8, 10, 12]
        
        if num in high_priority:
            return 'high-priority'
        elif num in medium_priority:
            return 'medium-priority'
        else:
            return 'low-priority'
    
    def create_issue(self, issue_data: Dict) -> bool:
        """Create a GitHub issue"""
        issue_body = f"""## Issue #{issue_data['number']}: {issue_data['title']}

### Description
{issue_data['description']}

### Phase
{issue_data['phase']}

### Priority
{issue_data['priority'].replace('-', ' ').title()}

### Acceptance Criteria
- [ ] Feature implemented according to specifications
- [ ] Code reviewed and approved
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Deployed to staging environment
- [ ] User acceptance testing completed

### Technical Notes
- This issue is part of the [RagaSense Development Roadmap](https://github.com/{REPO_OWNER}/{REPO_NAME}/blob/main/docs/ROADMAP.md)
- Please refer to the roadmap for context and dependencies
- Update progress in issue comments
- Link related pull requests to this issue

### Related Resources
- [Project Board](https://github.com/{REPO_OWNER}/{REPO_NAME}/projects)
- [Wiki Documentation](https://github.com/{REPO_OWNER}/{REPO_NAME}/wiki)
- [Contributing Guide](https://github.com/{REPO_OWNER}/{REPO_NAME}/blob/main/CONTRIBUTING.md)
"""
        
        issue_data_api = {
            'title': f"Issue #{issue_data['number']}: {issue_data['title']}",
            'body': issue_body,
            'labels': issue_data['labels'],
            'milestone': self.milestones.get(issue_data['phase'])
        }
        
        response = requests.post(
            f'{GITHUB_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/issues',
            headers=HEADERS,
            json=issue_data_api
        )
        
        if response.status_code == 201:
            issue = response.json()
            self.issues_created.append(issue)
            print(f"✅ Created Issue #{issue_data['number']}: {issue_data['title']}")
            print(f"   URL: {issue['html_url']}")
            return True
        else:
            print(f"❌ Failed to create Issue #{issue_data['number']}: {issue_data['title']}")
            print(f"Error: {response.text}")
            return False
    
    def run(self):
        """Main execution method"""
        if not GITHUB_TOKEN:
            print("❌ GITHUB_TOKEN environment variable not set")
            print("Please set your GitHub Personal Access Token:")
            print("export GITHUB_TOKEN='your_token_here'")
            return
        
        print("🚀 Starting RagaSense Roadmap Issue Creation...")
        print(f"Repository: {REPO_OWNER}/{REPO_NAME}")
        print()
        
        # Create milestones
        print("📅 Creating milestones...")
        self.create_milestones()
        print()
        
        # Create labels
        print("🏷️ Creating labels...")
        self.create_labels()
        print()
        
        # Parse roadmap
        print("📋 Parsing roadmap file...")
        issues = self.parse_roadmap_file()
        print(f"Found {len(issues)} issues to create")
        print()
        
        # Create issues
        print("🎯 Creating issues...")
        success_count = 0
        for issue_data in issues:
            if self.create_issue(issue_data):
                success_count += 1
            print()
        
        # Summary
        print("📊 Summary:")
        print(f"✅ Successfully created {success_count}/{len(issues)} issues")
        print(f"📅 Created {len(self.milestones)} milestones")
        print()
        print("🔗 Next steps:")
        print(f"1. View all issues: https://github.com/{REPO_OWNER}/{REPO_NAME}/issues")
        print(f"2. Set up project board: https://github.com/{REPO_OWNER}/{REPO_NAME}/projects")
        print(f"3. Review roadmap: https://github.com/{REPO_OWNER}/{REPO_NAME}/blob/main/docs/ROADMAP.md")
        print()
        print("🎉 Roadmap issues creation complete!")

if __name__ == '__main__':
    creator = RoadmapIssueCreator()
    creator.run()
