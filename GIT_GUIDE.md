# Git Guide for YOLOv5 Object Detection

```
yolov5-obj-detection/
├── .gitignore              # Git ignore rules
├── README.md               # Main documentation
├── SETUP_GUIDE.md          # Detailed setup instructions
├── GIT_GUIDE.md            # This file
├── requirements.txt        # Python dependencies
├── setup.ps1               # Windows setup script
├── setup.sh                # Linux/Mac setup script
├── data/
│   ├── dummy/
│   │   └── dataset.yaml    # Dummy dataset configuration
│   ├── fire_data/
│   │   ├── fire_dataset.yaml # Fire/smoke dataset configuration
│   │   └── preprocess_fire_dataset.py
└── scripts/
    ├── create_dummy_dataset.py
    ├── train_model.py
    ├── validate_model.py
    ├── train.ps1
    └── val.ps1
```


## Initial Setup for Git

### 1. Initialize Git Repository

If not already initialized:

```bash
git init
git add .
git commit -m "Initial commit: YOLOv5 object detection pipeline"
```

### 2. Create GitHub Repository

1. Go to [GitHub](https://github.com/new)
2. Create a new repository (e.g., `yolov5-obj-detection`)
3. **Do NOT** initialize with README (we already have one)

### 3. Push to GitHub

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/yolov5-obj-detection.git

# Push code
git branch -M main
git push -u origin main
```

## For Collaborators Cloning the Repo

### Initial Clone

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/yolov5-obj-detection.git
cd yolov5-obj-detection

# Run setup (automated)
# Windows:
.\setup.ps1
# Linux/Mac:
./setup.sh
```

The setup script will:
1. Clone YOLOv5 repository
2. Create virtual environment
3. Install all dependencies
4. Generate dummy dataset

### What Happens During Clone

**Downloaded (from your repo):**
- All scripts and documentation
- Dataset configuration templates
- Setup scripts

**NOT Downloaded (created during setup):**
- `.venv/` - Created by setup script
- `yolov5/` - Cloned during setup
- `data/dummy/` - Generated during setup

## Git Workflow

### Making Changes

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Test with dummy dataset
python scripts/train_model.py --epochs 3 --batch 4

# Check what changed
git status
git diff

# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add: description of your changes"

# Push to GitHub
git push origin feature/your-feature-name
```

### Updating from Main

```bash
# Switch to main branch
git checkout main

# Pull latest changes
git pull origin main

# Update your feature branch
git checkout feature/your-feature-name
git merge main
```

## Checking What Will Be Committed

Before committing, verify what files will be included:

```bash
# Show status
git status

# Show what will be committed
git diff --cached

# Show untracked files
git ls-files --others --exclude-standard

# Show ignored files (should include .venv, yolov5, etc.)
git status --ignored
```

## Common Git Commands

```bash
# View commit history
git log --oneline --graph

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard local changes
git restore <file>

# Create and switch to new branch
git checkout -b <branch-name>

# Delete local branch
git branch -d <branch-name>

# View remote URLs
git remote -v
```

## Repository Size Management

### Keep Repository Small

The `.gitignore` is configured to exclude large files:
- Model weights (*.pt files)
- Training outputs (runs/ folder)
- Dataset images
- Virtual environment

### If You Accidentally Commit Large Files

```bash
# Remove from Git but keep locally
git rm --cached <file>

# Remove from history (careful!)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch <file>" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (only if repository is not shared yet)
git push origin --force --all
```

## Setting Up Git LFS (Optional)

If you need to track model weights or large files:

```bash
# Install Git LFS
# Windows: Download from https://git-lfs.github.com/
# Linux: sudo apt install git-lfs
# Mac: brew install git-lfs

# Initialize Git LFS
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "*.pth"

# Commit .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for model weights"
```

## Creating a Release

When ready to release a version:

```bash
# Tag the release
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push tags to GitHub
git push origin v1.0.0

# Or push all tags
git push --tags
```

On GitHub, go to "Releases" and create a new release from the tag.

## Best Practices

1. **Commit Often**: Make small, logical commits
2. **Write Good Messages**: Be descriptive in commit messages
3. **Test Before Committing**: Run the pipeline with dummy dataset
4. **Don't Commit Secrets**: API keys, passwords, credentials
5. **Keep .gitignore Updated**: Add new patterns as needed
6. **Use Branches**: Don't commit directly to main
7. **Review Before Push**: Use `git diff` to review changes

## Troubleshooting

### Issue: Large files rejected by GitHub

**Solution:**
Check if large files are being tracked:
```bash
git ls-files | xargs du -h | sort -h | tail -20
```

Add them to `.gitignore` and remove from Git:
```bash
echo "*.pt" >> .gitignore
git rm --cached *.pt
git commit -m "Remove model weights from tracking"
```

### Issue: Merge conflicts

**Solution:**
```bash
# View conflicting files
git status

# Edit files to resolve conflicts
# ... resolve conflicts ...

# Mark as resolved
git add <file>

# Complete merge
git commit
```

### Issue: Wrong files committed

**Solution:**
```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Update .gitignore
echo "file-to-ignore" >> .gitignore

# Recommit correctly
git add .
git commit -m "Fixed: correct files"
```

## Example Workflow

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/yolov5-obj-detection.git
cd yolov5-obj-detection
./setup.sh  # or setup.ps1 on Windows

# Make changes
git checkout -b feature/add-preprocessing
# ... edit scripts/preprocess_dataset.py ...

# Test
python scripts/create_dummy_dataset.py
python scripts/train_model.py --epochs 3 --batch 4

# Commit
git add scripts/preprocess_dataset.py
git commit -m "Add: preprocessing script for custom datasets"

# Push
git push origin feature/add-preprocessing

# Create pull request on GitHub
```

## Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [Conventional Commits](https://www.conventionalcommits.org/)
