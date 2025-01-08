# BDC800-Capstone
Repository for "BDC800 - Capstone Project" in Seneca Polytechnic: RAG retrieval system

## 👥 Collaborators

- [Ran Arino](https://github.com/RanArino)
- [Nipun Ravindu Fernando Warnakulasuriya]()
- [Umer Aftab]()

## 🚀� Resources

### Project Documentation
- [Google Drive: BDC800 - Capstone](https://drive.google.com/drive/folders/1jclLloZ07zzy-dRU02OAHqAbOkTywCYA?usp=sharing) - Project documentation storage
  - Project Proposal
  - Final Paper
  - Presentation Slides
  - Other submission materials

### Knowledge Management
- [Notion: Capstone Project Page](https://www.notion.so/Capstone-Project-Page-1755dbaba5c080dbae60fbb4eff2f8bf?pvs=4) - Literature database & note-taking
  - Research papers and articles
  - Meeting notes
  - Project progress tracking
  - Team collaboration workspace

## 🚀 Getting Started

### Clone Repository

#### Using VS Code
1. Open VS Code
2. Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac) to open command palette
3. Type "Git: Clone" and select it
4. Paste the repository URL: `https://github.com/RanArino/BDC800-Capstone.git`
5. Choose a local directory for the project

#### Using Cursor
1. Open Cursor
2. Click on the "Source Control" icon in the left sidebar
3. Click "Clone Repository"
4. Paste the repository URL: `https://github.com/RanArino/BDC800-Capstone.git`
5. Choose a local directory for the project

### Set Up Virtual Environment

#### Windows
```bash
# Navigate to project directory
cd BDC800-Capstone

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### macOS/Linux
```bash
# Navigate to project directory
cd BDC800-Capstone

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

To deactivate the virtual environment when you're done:
```bash
deactivate
```

## 🌿 Git Workflow for Team Members

### Initial Setup (One Time Only)
1. After cloning, make sure you're on the main branch:
```bash
git checkout main
git pull origin main
```

2. Configure your Git identity:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Starting Your Work

#### 1. Create Your Feature Branch
Always create a new branch for each feature/task:
```bash
# Get latest changes from main branch
git checkout main
git pull origin main

# Create and switch to your new branch
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- For new features: `feature/feature-name`
- For bug fixes: `fix/bug-name`
- For documentation: `docs/what-you-are-documenting`

Example:
```bash
git checkout -b feature/login-page
```

#### 2. Working on Your Branch
1. Make your changes to the code
2. Regularly commit your changes:
```bash
# Check what files you've changed
git status

# Add your changes
git add .   # To add all changes
# OR
git add filename.py   # To add specific files

# Commit your changes
git commit -m "Brief description of your changes"
```

#### 3. Keeping Your Branch Updated
Regularly sync with main branch to avoid conflicts:
```bash
# Save your current changes
git stash

# Update main
git checkout main
git pull origin main

# Return to your branch
git checkout feature/your-feature-name
git merge main

# Restore your changes
git stash pop
```

#### 4. Pushing Your Changes
Push your branch to GitHub:
```bash
git push origin feature/your-feature-name
```

#### 5. Creating a Pull Request (PR)
1. Go to the repository on GitHub
2. Click on "Pull requests"
3. Click "New pull request"
4. Select your branch to merge into main
5. Fill in the PR template:
   - Title: Brief description of your changes
   - Description: Detailed explanation of what you did
   - Add any relevant screenshots
6. Request review from team members
7. Wait for approval before merging

### 🚫 Common Mistakes to Avoid
1. Never commit directly to the main branch
2. Don't merge your own PR without review
3. Keep commits focused and small
4. Write clear commit messages
5. Always pull before starting new work

