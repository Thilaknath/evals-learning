# LLM Evals Learning

A project for learning LLM evaluation techniques using LangChain and OpenAI.

## Overview

This repository contains a quiz generation assistant powered by GPT-3.5-turbo. The assistant creates quizzes about Art, Science, and Geography based on a predefined quiz bank. The project demonstrates how to build and evaluate LLM-powered applications.

## Project Structure

```
├── app.py                 # Quiz assistant implementation using LangChain
├── test_assistant.py      # Pytest evaluations for the assistant
├── save_eval_artifacts.py # Script to generate HTML evaluation reports
├── requirements.txt       # Python dependencies
└── .github/workflows/
    └── evals.yml          # GitHub Actions workflow for CI/CD
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up OpenAI API Key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

Or export it as an environment variable:

```bash
export OPENAI_API_KEY=your-api-key-here
```

## Running Locally

### Run evaluations

```bash
python -m pytest test_assistant.py
```

### Generate evaluation report

```bash
python save_eval_artifacts.py
```

## CI/CD with GitHub Actions

The project uses GitHub Actions for continuous evaluation. Three modes are available:

| Mode | Trigger | Description |
|------|---------|-------------|
| `commit` | Push to main/master or manual | Runs commit-level evaluations |
| `full` | Manual only | Runs full evaluation suite |
| `report` | Manual only | Generates HTML evaluation report |

### Setting up GitHub Actions

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `OPENAI_API_KEY`
5. Value: Your OpenAI API key

### Running workflows manually

1. Go to **Actions** tab in your repository
2. Select **LLM Evals** workflow
3. Click **Run workflow**
4. Select the evaluation mode (`commit`, `full`, or `report`)
5. Click **Run workflow**

### Viewing results

- Test results are uploaded as artifacts after each run
- Download from the workflow run summary page
- HTML reports are available when running in `report` mode

## License

MIT
