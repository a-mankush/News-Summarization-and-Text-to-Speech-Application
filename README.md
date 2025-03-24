# Project Setup

This document provides step-by-step instructions to install and run the application.

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)
- Virtual environment (optional but recommended)
- Hugging Face API access

## Installation

### 1. Clone the Repository
```sh
$ git clone <repository-url>
$ cd <project-directory>
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)
#### On macOS/Linux:
```sh
$ python3 -m venv venv
$ source venv/bin/activate
```
#### On Windows:
```sh
$ python -m venv venv
$ venv\Scripts\activate
```

### 3. Install Required Dependencies
```sh
$ pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root and add necessary credentials:
```
HUGGINGFACE_API_KEY=<your-api-key>
```

## Running the Application

### 1. Execute the Main Script
```sh
$ python main.py
```

This will:
- Fetch articles related to the specified company (`COMPANY` variable in `main.py`)
- Analyze sentiment and coverage differences
- Generate a final sentiment report
- Translate the report to Hindi
- Generate Hindi TTS (Text-to-Speech) output

### 2. Output
- The final sentiment analysis will be printed in English and Hindi.
- An audio file `final_sentiment_hindi.mp3` will be generated containing the Hindi-translated analysis.

## File Overview
- `main.py` - Orchestrates the article fetching, sentiment analysis, and final output generation.
- `schema.py` - Defines data models using Pydantic.
- `templates.py` - Contains prompt templates for generating structured responses.
- `utils.py` - Includes helper functions for fetching articles, translating text, and generating TTS output.

## Troubleshooting
- If the script fails to fetch articles, ensure your internet connection is active.
- If `Hugging Face API` errors occur, verify that your API key is valid.
- Ensure `gtts` is installed for TTS functionality.

