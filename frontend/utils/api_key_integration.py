"""
Integration utilities for API Key Manager with backend pipeline
"""

import os
from typing import Dict, List, Optional
import streamlit as st


def update_environment_with_keys(api_keys: Dict[str, str]) -> None:
    """
    Update environment variables with validated API keys
    
    Args:
        api_keys: Dictionary of API keys from the API Key Manager
    """
    for key, value in api_keys.items():
        if value:
            os.environ[key] = value


def get_available_models(api_keys: Dict[str, str]) -> List[str]:
    """
    Get list of available models based on provided API keys
    
    Args:
        api_keys: Dictionary of API keys from the API Key Manager
    
    Returns:
        List of model names available for use
    """
    model_mapping = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Claude",
        "LLAMA_API_KEY": "Llama",
        "XAI_API_KEY": "Grok",
        "DEEPSEEK_API_KEY": "DeepSeek"
    }
    
    available = []
    for key, model_name in model_mapping.items():
        if key in api_keys and api_keys[key]:
            available.append(model_name)
    
    return available


def validate_minimum_keys(api_keys: Dict[str, str], required_count: int = 1) -> bool:
    """
    Validate that minimum number of API keys are provided
    
    Args:
        api_keys: Dictionary of API keys from the API Key Manager
        required_count: Minimum number of keys required
    
    Returns:
        True if minimum requirements are met
    """
    return len(api_keys) >= required_count


def create_model_config_override(api_keys: Dict[str, str]) -> Dict:
    """
    Create a MODEL_CONFIG override dictionary with provided API keys
    
    Args:
        api_keys: Dictionary of API keys from the API Key Manager
    
    Returns:
        Dictionary compatible with backend MODEL_CONFIG format
    """
    config_override = {}
    
    if "OPENAI_API_KEY" in api_keys:
        config_override["OpenAI"] = {
            "client": "openai",
            "model": "gpt-4o",
            "api_key": api_keys["OPENAI_API_KEY"]
        }
    
    if "ANTHROPIC_API_KEY" in api_keys:
        config_override["Claude"] = {
            "client": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "api_key": api_keys["ANTHROPIC_API_KEY"]
        }
    
    if "LLAMA_API_KEY" in api_keys:
        config_override["Llama"] = {
            "client": "llamaapi",
            "model": "llama3.1-70b",
            "api_key": api_keys["LLAMA_API_KEY"]
        }
    
    if "XAI_API_KEY" in api_keys:
        config_override["Grok"] = {
            "client": "openai",
            "model": "grok-2-latest",
            "api_key": api_keys["XAI_API_KEY"],
            "base_url": "https://api.x.ai/v1"
        }
    
    if "DEEPSEEK_API_KEY" in api_keys:
        config_override["DeepSeek"] = {
            "client": "llamaapi",
            "model": "deepseek-v3",
            "api_key": api_keys["DEEPSEEK_API_KEY"]
        }
    
    return config_override


def render_model_selector(api_keys: Dict[str, str]) -> List[str]:
    """
    Render a model selection widget based on available API keys
    
    Args:
        api_keys: Dictionary of API keys from the API Key Manager
    
    Returns:
        List of selected model names
    """
    available_models = get_available_models(api_keys)
    
    if not available_models:
        st.warning("No models available. Please configure API keys first.")
        return []
    
    selected = st.multiselect(
        "Select Models to Survey",
        options=available_models,
        default=available_models[:1],  # Default to first available
        help="Choose which models to include in the survey"
    )
    
    return selected


def check_api_costs_warning(selected_models: List[str], num_questions: int, num_runs: int) -> None:
    """
    Display a warning about potential API costs
    
    Args:
        selected_models: List of selected model names
        num_questions: Number of survey questions
        num_runs: Number of runs per model
    """
    # Rough cost estimates per 1000 tokens (very approximate)
    cost_estimates = {
        "OpenAI": 0.03,  # GPT-4
        "Claude": 0.025,  # Claude 3.5
        "Llama": 0.001,  # Llama API
        "Grok": 0.02,  # X.AI
        "DeepSeek": 0.002  # DeepSeek
    }
    
    # Estimate tokens (very rough)
    estimated_tokens_per_question = 200  # Prompt + response
    total_api_calls = len(selected_models) * num_questions * num_runs
    total_estimated_tokens = total_api_calls * estimated_tokens_per_question
    
    # Calculate estimated cost
    total_cost = 0
    for model in selected_models:
        if model in cost_estimates:
            model_cost = (total_estimated_tokens / 1000) * cost_estimates[model]
            total_cost += model_cost
    
    if total_cost > 1.0:  # Warn if estimated cost > $1
        st.warning(f"""
        ⚠️ **Estimated API Costs**: ~${total_cost:.2f}
        
        - Total API calls: {total_api_calls:,}
        - Estimated tokens: {total_estimated_tokens:,}
        
        *Note: This is a rough estimate. Actual costs may vary.*
        """)
    else:
        st.info(f"""
        💰 Estimated API costs: ~${total_cost:.2f} 
        ({total_api_calls:,} API calls)
        """)


def save_api_keys_to_session(api_keys: Dict[str, str]) -> None:
    """
    Save API keys to Streamlit session state for persistence across pages
    
    Args:
        api_keys: Dictionary of API keys to save
    """
    st.session_state.stored_api_keys = api_keys


def load_api_keys_from_session() -> Dict[str, str]:
    """
    Load API keys from Streamlit session state
    
    Returns:
        Dictionary of API keys or empty dict if none stored
    """
    return st.session_state.get("stored_api_keys", {})


def clear_api_keys_from_session() -> None:
    """
    Clear API keys from session state
    """
    if "stored_api_keys" in st.session_state:
        del st.session_state.stored_api_keys
    if "api_keys" in st.session_state:
        del st.session_state.api_keys
    if "api_key_validation" in st.session_state:
        del st.session_state.api_key_validation