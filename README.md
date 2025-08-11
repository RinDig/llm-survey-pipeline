# 🔬 LLM Survey Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llm-survey-pipeline-wwfz4ksugwzfssnzslviyk.streamlit.app/))
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Live App**: [https://llm-survey-pipeline.streamlit.app](https://llm-survey-pipeline.streamlit.app)

## Overview

A comprehensive research platform for conducting psychological and political surveys with Large Language Models (LLMs). This tool systematically tests model responses across established psychological scales under different prompting conditions, providing researchers with powerful tools to analyze potential biases, personality traits, or cognitive styles embedded in these models.

![WhatsApp Image 2025-06-10 at 17 42 25_797f3260](https://github.com/user-attachments/assets/c79330df-9e96-4f25-a396-8135dcce1f2f)

## ✨ Key Features

- **🌐 Web-Based Interface**: No installation required - use directly in your browser
- **🔐 Secure API Key Management**: Keys stored only in browser session, never persisted to servers
- **🤖 Multi-Model Support**: Test OpenAI, Claude, Llama, Grok, and DeepSeek simultaneously
- **📊 Built-in Psychological Scales**: RWA, LWA, MFQ, and NFC scales ready to use
- **✏️ Custom Prompt Builder**: Create and test your own prompting personas with live preview
- **📈 Real-time Progress Tracking**: Monitor execution with detailed progress updates
- **💾 Flexible Data Export**: Download results in CSV, JSON, or Excel formats
- **🔍 Advanced Data Explorer**: Filter, visualize, and analyze results interactively
- **💰 Cost Tracking**: Automatic token usage and cost estimation per model

## Directory Structure

```
llm_survey_pipeline/
├── config/              # Configuration modules
│   ├── models.py        # Model configurations
│   ├── prompts.py       # Prompt templates (preserved exactly)
│   └── scales.py        # Scale templates (RWA, LWA, MFQ, NFC - preserved exactly)
├── core/                # Core functionality
│   ├── api_clients.py   # API client implementations
│   ├── parsers.py       # Response parsing logic
│   ├── processors.py    # Task processing engine
│   └── validators.py    # Data validation
├── models/              # Data models
│   └── data_models.py   # Pydantic models
├── utils/               # Utilities
│   ├── cost_tracking.py # Token/cost tracking
│   └── analysis.py      # Analysis functions
├── dashboard/           # Web dashboard
│   └── app.py          # Streamlit application
├── data/outputs/        # Output directory
├── main.py             # CLI interface
└── requirements.txt
```

## 🚀 Quick Start

### Option 1: Use the Live Web App (Recommended)

1. Visit [(https://llm-survey-pipeline-wwfz4ksugwzfssnzslviyk.streamlit.app/)]
2. Enter your API keys in the Setup page
3. Configure your survey parameters
4. Execute and download results

No installation required!

### Option 2: Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RinDig/llm-survey-pipeline.git
   cd llm-survey-pipeline
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Open your browser to `http://localhost:8501`

## Usage

### Using the Web Interface

1. **🔑 Setup API Keys**:
   - Navigate to the "Setup" page
   - Enter API keys for the models you want to test
   - Use "Test All Keys" to validate all at once

2. **📝 Configure Survey**:
   - Select models to test
   - Choose psychological scales
   - Pick or create custom prompt styles
   - Set temperature and number of runs

3. **🚀 Execute Survey**:
   - Review configuration and cost estimates
   - Click "Start Survey" to begin
   - Monitor real-time progress

4. **📊 Analyze Results**:
   - View completed surveys in the Results page
   - Use Data Explorer for advanced filtering
   - Export in your preferred format

### Command Line Interface

For automated runs or integration with scripts:

```bash
python main.py --scales RWA LWA --models OpenAI Claude --prompts minimal extreme_liberal --runs 2
```

Options:
- `--scales`: List of scales to run (RWA, RWA2, LWA, MFQ, NFC)
- `--models`: List of models to test (OpenAI, Claude, Grok, Llama, DeepSeek)
- `--prompts`: Prompt styles to use
- `--runs`: Number of runs per question
- `--temperature`: Model temperature (default: 0.0)
- `--output-dir`: Output directory for results

## Adding New Components

### Adding a New Scale

1. Edit `config/scales.py`
2. Add your scale questions following the existing format:
   ```python
   new_scale_questions = [
       {"scale_name": "NEW", "id": "NEW_1", "text": "Question text", 
        "scale_range": [1,7], "reverse_score": False},
       # ... more questions
   ]
   ```
3. Add to `all_scales` list

### Adding a New Model

1. Edit `config/models.py`
2. Add model configuration:
   ```python
   "NewModel": {
       "client": "openai",  # or "anthropic", "llamaapi"
       "model": "model-name",
       "api_key": os.getenv("NEW_MODEL_API_KEY"),
   }
   ```

### Adding a New Prompt Style

1. Edit `config/prompts.py`
2. Add prompt template:
   ```python
   "new_style": "Your prompt template here..."
   ```

## 🔑 Getting API Keys

### OpenAI
- Visit [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Create a new API key
- Requires payment method on file

### Anthropic Claude  
- Visit [https://console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
- Generate an API key
- Requires approved account with credits

### X.AI (Grok)
- Visit [https://x.ai/api](https://x.ai/api)
- Request API access
- Currently in beta with limited availability

### Llama
- Visit [https://www.llama-api.com](https://www.llama-api.com)
- Sign up for free tier or paid plan
- Generate API token

### DeepSeek
- Visit [https://platform.deepseek.com](https://platform.deepseek.com)
- Create account and add credits
- Generate API key

## 📊 Psychological Scales

- **RWA (Right-Wing Authoritarianism)**: Measures authoritarian personality traits
- **LWA (Left-Wing Authoritarianism)**: Assesses left-leaning authoritarian traits
- **MFQ (Moral Foundations Questionnaire)**: Evaluates moral reasoning across 5 foundations
- **NFC (Need for Closure)**: Measures preference for order and decisiveness

## 📁 Output Files

- `unified_responses.csv`: All survey responses with scores
- `refusal_responses.csv`: Responses where models refused/failed
- `mfq_foundation_scores.csv`: MFQ foundation analysis (if MFQ scale is run)

## ⚡ Performance & Rate Limits

The system includes intelligent rate limiting:
- **OpenAI/Grok**: 3 concurrent calls, 1 second between chunks
- **Anthropic**: 5 concurrent calls, 0.5 seconds between chunks
- **Llama/DeepSeek**: 10 concurrent calls, 0.2 seconds between chunks

## 🔒 Security & Privacy

- API keys are stored only in your browser session
- No data is sent to external servers (except LLM API calls)
- All processing happens client-side or on your local machine
- Survey results remain private to your session

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Psychological scales adapted from validated research instruments
- Thanks to all LLM providers for API access

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on [GitHub](https://github.com/RinDig/llm-survey-pipeline/issues)
- Check existing issues for solutions

---

**Note**: Ensure you comply with each LLM provider's terms of service and have appropriate permissions for research use.
