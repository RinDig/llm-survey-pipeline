# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python pipeline for administering psychological and political surveys to various Large Language Models (LLMs). It tests model responses under different prompting conditions across established psychological scales (RWA, LWA, MFQ, NFC).

## Development Commands

### Setup and Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

**Web Dashboard (Interactive):**
```bash
streamlit run dashboard/app.py
# Or use the launch script:
python run_dashboard.py
```

**CLI Interface (Automated):**
```bash
python main.py --scales RWA LWA --models OpenAI Claude --prompts minimal extreme_liberal --runs 2
```

### Environment Setup
Create a `.env` file with API keys:
```
OPENAI_API_KEY="your-key"
ANTHROPIC_API_KEY="your-key"  
LLAMA_API_KEY="your-key"
XAI_API_KEY="your-key"
DEEPSEEK_API_KEY="your-key"
```

## Architecture

### Module Structure
- **config/**: Configuration modules
  - `models.py`: LLM model configurations and API keys
  - `prompts.py`: Prompt templates for different political personas
  - `scales.py`: Psychological scale questions (RWA, LWA, MFQ, NFC)
  
- **core/**: Core processing engine
  - `api_clients.py`: Unified API client for all LLM providers
  - `processors.py`: Task processing with rate limiting and concurrency control
  - `parsers.py`: Response parsing and validation
  - `validators.py`: Data validation logic

- **dashboard/**: Streamlit web interface
  - `app.py`: Interactive dashboard for configuration and visualization

- **utils/**: Utility functions
  - `cost_tracking.py`: Token usage and cost tracking
  - `analysis.py`: Statistical analysis and MFQ foundation scoring

### Key Design Patterns

1. **Async Processing**: Uses asyncio for concurrent API calls with provider-specific rate limiting:
   - OpenAI/Grok: 3 concurrent calls, 1 second between chunks
   - Anthropic: 5 concurrent calls, 0.5 seconds between chunks  
   - Llama/DeepSeek: 10 concurrent calls, 0.2 seconds between chunks

2. **Modular Configuration**: All scales, prompts, and models are configured in separate modules for easy extension.

3. **Data Flow**:
   - Tasks are built from combinations of scales × models × prompts × runs
   - Processed asynchronously with progress tracking
   - Results saved to CSV with reverse scoring applied where needed
   - Refusal responses tracked separately

### Output Files
- `data/outputs/unified_responses.csv`: All survey responses with scores
- `data/outputs/refusal_responses.csv`: Failed/refused responses
- `data/outputs/mfq_foundation_scores.csv`: MFQ foundation analysis (when MFQ scale is run)

## Adding New Components

### New Scale
Add to `config/scales.py` following the existing format with `scale_name`, `id`, `text`, `scale_range`, and `reverse_score` fields.

### New Model  
Add to `config/models.py` MODEL_CONFIG dictionary with `client`, `model`, `api_key`, and optional `base_url`.

### New Prompt Style
Add to `config/prompts.py` prompt_templates dictionary.

## Important Notes

- No test suite currently exists
- Uses module imports like `from llm_survey_pipeline.config import ...`
- Dashboard requires `PYTHONPATH` setup (handled by `run_dashboard.py`)
- All API calls include retry logic and error handling
- Token usage tracked per model in `cost_tracker` dictionary