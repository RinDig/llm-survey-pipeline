"""
Demo application showing how to use the API Key Manager component
Run with: streamlit run frontend/demo_api_keys.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.components import render_api_key_manager

def main():
    st.set_page_config(
        page_title="LLM Survey Pipeline - API Key Setup",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stButton button {
        height: 2.5rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render the API key manager
    validated_keys = render_api_key_manager()
    
    # Additional demo features
    st.divider()
    
    # Show how to use the validated keys
    with st.expander("📝 Integration Example", expanded=False):
        st.markdown("""
        ### How to use validated keys in your pipeline:
        
        ```python
        from frontend.components import render_api_key_manager
        
        # In your Streamlit app
        validated_keys = render_api_key_manager()
        
        # Pass to backend survey pipeline
        if validated_keys:
            # Keys are in format expected by MODEL_CONFIG
            # e.g., {'OPENAI_API_KEY': 'sk-...', 'ANTHROPIC_API_KEY': 'sk-...'}
            
            # Update environment or pass directly to pipeline
            for key, value in validated_keys.items():
                os.environ[key] = value
            
            # Now run your survey
            run_survey(models=selected_models, api_keys=validated_keys)
        ```
        """)
    
    # Show current configuration
    if validated_keys:
        with st.expander("✅ Ready to Run Survey", expanded=True):
            st.success("Your API keys are configured and validated!")
            
            # Show which models are available
            model_mapping = {
                "OPENAI_API_KEY": "OpenAI (GPT-4)",
                "ANTHROPIC_API_KEY": "Claude",
                "LLAMA_API_KEY": "Llama 3.1",
                "XAI_API_KEY": "Grok",
                "DEEPSEEK_API_KEY": "DeepSeek V3"
            }
            
            available_models = []
            for key in validated_keys:
                if key in model_mapping:
                    available_models.append(model_mapping[key])
            
            if available_models:
                st.write("**Available Models:**")
                for model in available_models:
                    st.write(f"- {model}")
            
            # Example next steps
            st.info("""
            **Next Steps:**
            1. Select survey scales (RWA, LWA, MFQ, NFC)
            2. Choose prompt styles (minimal, moderate, extreme)
            3. Configure number of runs
            4. Start the survey pipeline
            """)
    
    # Footer with helpful links
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📚 Documentation")
        st.markdown("""
        - [OpenAI Docs](https://platform.openai.com/docs)
        - [Anthropic Docs](https://docs.anthropic.com)
        - [X.AI Docs](https://docs.x.ai)
        """)
    
    with col2:
        st.markdown("### 💰 Pricing")
        st.markdown("""
        - [OpenAI Pricing](https://openai.com/pricing)
        - [Anthropic Pricing](https://anthropic.com/pricing)
        - [DeepSeek Pricing](https://platform.deepseek.com/pricing)
        """)
    
    with col3:
        st.markdown("### 🛡️ Security")
        st.markdown("""
        - Keys stored in session only
        - Never persisted to disk
        - Cleared on browser close
        """)


if __name__ == "__main__":
    main()