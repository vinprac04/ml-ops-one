# ml-ops-assignment-one

## Project Setup

This document captures the steps followed to set up the local environment, install dependencies, configure GitHub SSH access, and connect the repository.

---

# 1. Miniconda Installation & Environment Setup

## Step 1 – Download Miniconda

Download Miniconda installer:

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
```

---

## Step 2 – Install Miniconda

Run:

```bash
bash ./Miniconda3-latest-MacOSX-arm64.sh
```

Installation path used:

```bash
/users/vinaysingh/claude-projects/miniconda3
```

After install, restart terminal.

---

## Step 3 – Create Conda Environment

Create environment with Python 3.12:

```bash
conda create -n mlOpsOne python=3.12
```

---

## Step 4 – Activate Environment

```bash
conda activate mlOpsOne
```

---

## Step 5 – requirements.txt Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### requirements.txt

```txt
pandas==2.3.2
matplotlib==3.10.6
scikit-learn==1.7.2
numpy==2.3.2
jupyterlab==4.4.6
ipykernel==6.30.1
```

If needed, regenerate requirements file using:

```bash
pip freeze > requirements.txt
```

---

# 2. GitHub SSH Setup

SSH is used to securely connect local machine with GitHub so code can be pushed without username/password each time.

---

## Step 1 – Generate SSH Key

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```

Press Enter to save in default location:

```bash
~/.ssh/id_ed25519
```

---

## Step 2 – Start SSH Agent

```bash
eval "$(ssh-agent -s)"
```

---

## Step 3 – Add SSH Key

```bash
ssh-add ~/.ssh/id_ed25519
```

---

## Step 4 – Copy Public Key

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the output.

---

## Step 5 – Add Key in GitHub

In [GitHub](https://github.com?utm_source=chatgpt.com):

* Open **Settings**
* Go to **SSH and GPG Keys**
* Click **New SSH Key**
* Paste copied key
* Save

---

## Step 6 – Verify SSH Connection

```bash
ssh -T git@github.com
```

Expected response:

```bash
Hi vinprac04! You've successfully authenticated...
```

---

# 3. Git Repository Setup

## Clone repository

```bash
git clone git@github.com:vinprac04/ml-ops-one.git
```

Move into project folder:

```bash
cd ml-ops-assignment-one
```

---

## Initialize git

```bash
git init
```

---

## Add remote origin

```bash
git remote add origin git@github.com:vinprac04/ml-ops-one.git
```

---

## Verify remote

```bash
git remote -v
```

---

## Commit and Push

```bash
git add .
git commit -m "Initial commit"
git branch -M main
git push -u origin main
```

---

# 4. Run Project

Run:

```bash
python main.py
```

If using notebook:

```bash
jupyter lab
```

---

# Project Structure

```bash
ml-ops-assignment-one/
│
├── README.md
├── requirements.txt
└── source files
```
