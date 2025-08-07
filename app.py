"""
LLM Survey Pipeline - Research Dashboard
A comprehensive tool for conducting psychological surveys with Large Language Models
"""
import streamlit as st
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Import components
from frontend.components.api_key_manager import render_api_key_manager
from frontend.components.prompt_builder import render_prompt_builder
from frontend.components.survey_executor import SurveyExecutor
from frontend.components.data_explorer import DataExplorer
from backend.storage.json_handler import StorageManager

# Page configuration
st.set_page_config(
    page_title="LLM Survey Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .step-header {
        background: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}
    if 'selected_prompts' not in st.session_state:
        st.session_state.selected_prompts = {}
    if 'survey_config' not in st.session_state:
        st.session_state.survey_config = {}
    if 'survey_results' not in st.session_state:
        st.session_state.survey_results = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Home"

def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 LLM Survey Pipeline</h1>
        <p>Conduct psychological surveys with Large Language Models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 📍 Navigation")
        st.caption("*Click any option below to navigate*")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🔑 Setup", "📝 Configure Survey", "🚀 Execute", "📊 Results", "🔍 Data Explorer", "📚 Documentation"],
            label_visibility="collapsed"
        )
        
        # Show workflow progress
        st.markdown("---")
        st.markdown("### 📋 Workflow Status")
        
        # Check progress
        has_keys = bool(st.session_state.api_keys)
        has_config = bool(st.session_state.survey_config.get('models')) if 'survey_config' in st.session_state else False
        
        if has_keys:
            st.success("✅ API Keys configured")
        else:
            st.info("⏳ API Keys needed")
            
        if has_config:
            st.success("✅ Survey configured")
        else:
            st.info("⏳ Survey configuration needed")
        
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        storage = StorageManager()
        recent_runs = storage.get_recent_runs(limit=5)
        st.metric("Total Surveys", len(recent_runs))
        
        if recent_runs:
            st.markdown("**Recent Runs:**")
            for run in recent_runs[:3]:
                st.text(f"• {run['timestamp'][:10]}")
                st.caption(f"  {', '.join(run['summary']['models_used'][:2])}")
    
    # Main content area
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔑 Setup":
        show_setup_page()
    elif page == "📝 Configure Survey":
        show_configure_page()
    elif page == "🚀 Execute":
        show_execute_page()
    elif page == "📊 Results":
        show_results_page()
    elif page == "🔍 Data Explorer":
        show_explorer_page()
    elif page == "📚 Documentation":
        show_documentation_page()

def show_home_page():
    """Home page with overview and quick start"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Welcome to LLM Survey Pipeline")
        
        # Add a prominent callout box
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 10px; border-left: 4px solid #667eea;'>
            <h4>👈 Use the Navigation Menu to Get Started</h4>
            <p>All features are accessible through the sidebar navigation on the left.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")  # Add spacing
        
        st.markdown("""
        This platform enables researchers to conduct psychological and political surveys with various 
        Large Language Models (LLMs) under different prompting conditions.
        
        ### 🎯 Key Features
        - **Multiple LLM Support**: OpenAI, Anthropic, Llama, Grok, DeepSeek
        - **Psychological Scales**: RWA, LWA, MFQ, NFC
        - **Custom Prompts**: Build and test your own prompt templates
        - **Comprehensive Analysis**: Statistical tools and visualizations
        - **Secure**: API keys never leave your browser session
        """)
        
        st.markdown("### 🚀 Getting Started Guide")
        
        # Create a visual workflow
        col_1, col_2, col_3, col_4 = st.columns(4)
        
        with col_1:
            st.markdown("""
            <div style='text-align: center; padding: 1rem; background: #e3f2fd; border-radius: 10px; height: 120px;'>
                <h3>1️⃣</h3>
                <b>Setup</b><br>
                Add API keys
            </div>
            """, unsafe_allow_html=True)
        
        with col_2:
            st.markdown("""
            <div style='text-align: center; padding: 1rem; background: #e8f5e9; border-radius: 10px; height: 120px;'>
                <h3>2️⃣</h3>
                <b>Configure</b><br>
                Select options
            </div>
            """, unsafe_allow_html=True)
        
        with col_3:
            st.markdown("""
            <div style='text-align: center; padding: 1rem; background: #fff3e0; border-radius: 10px; height: 120px;'>
                <h3>3️⃣</h3>
                <b>Execute</b><br>
                Run survey
            </div>
            """, unsafe_allow_html=True)
        
        with col_4:
            st.markdown("""
            <div style='text-align: center; padding: 1rem; background: #f3e5f5; border-radius: 10px; height: 120px;'>
                <h3>4️⃣</h3>
                <b>Analyze</b><br>
                View results
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Platform Statistics")
        storage = StorageManager()
        all_runs = storage.get_recent_runs(limit=100)
        
        if all_runs:
            total_responses = sum(run['summary'].get('total_responses', 0) for run in all_runs)
            unique_models = set()
            for run in all_runs:
                unique_models.update(run['summary'].get('models_used', []))
            
            st.metric("Total Surveys", len(all_runs))
            st.metric("Total Responses", f"{total_responses:,}")
            st.metric("Models Tested", len(unique_models))
        else:
            st.info("No surveys conducted yet")
        
        st.markdown("### 📚 Resources")
        st.markdown("""
        - [GitHub Repository](https://github.com/yourusername/llm-survey-pipeline)
        - [Documentation](https://docs.llm-survey.com)
        - [Report Issues](https://github.com/yourusername/llm-survey-pipeline/issues)
        """)

def show_setup_page():
    """Setup page for API keys"""
    st.markdown("## 🔑 API Key Setup")
    st.markdown("""
    <div class="info-box">
    <b>Security Note:</b> Your API keys are stored only in your browser session and are never sent to our servers.
    </div>
    """, unsafe_allow_html=True)
    
    # Render API key manager
    api_keys = render_api_key_manager()
    
    if api_keys:
        st.session_state.api_keys = api_keys
        st.success(f"✅ {len(api_keys)} API keys configured")
        
        st.info("👈 **Next step: Select '📝 Configure Survey' from the navigation menu**")

def show_configure_page():
    """Configuration page for survey setup"""
    st.markdown("## 📝 Survey Configuration")
    
    tabs = st.tabs(["📋 Basic Settings", "✏️ Custom Prompts", "⚙️ Advanced"])
    
    with tabs[0]:
        st.markdown("### Select Survey Components")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model selection
            st.markdown("#### Models")
            
            # Check if any API keys are configured
            if not st.session_state.api_keys:
                st.warning("⚠️ No API keys configured yet!")
                st.info("👈 **Please select '🔑 Setup' from the navigation menu to add your API keys first**")
                st.caption("You need to enter API keys to see available models here")
            else:
                # Build list of available models based on configured keys
                available_models = []
                configured_keys = []
                
                if st.session_state.api_keys.get("OPENAI_API_KEY"):
                    available_models.append("OpenAI")
                    configured_keys.append("OpenAI")
                if st.session_state.api_keys.get("ANTHROPIC_API_KEY"):
                    available_models.append("Claude")
                    configured_keys.append("Anthropic")
                if st.session_state.api_keys.get("LLAMA_API_KEY"):
                    available_models.append("Llama")
                    configured_keys.append("Llama")
                if st.session_state.api_keys.get("XAI_API_KEY"):
                    available_models.append("Grok")
                    configured_keys.append("X.AI/Grok")
                if st.session_state.api_keys.get("DEEPSEEK_API_KEY"):
                    available_models.append("DeepSeek")
                    configured_keys.append("DeepSeek")
                
                if available_models:
                    st.caption(f"✅ API keys found for: {', '.join(configured_keys)}")
                    selected_models = st.multiselect(
                        "Select models to test",
                        available_models,
                        default=available_models[:2] if len(available_models) >= 2 else available_models
                    )
                    st.session_state.survey_config['models'] = selected_models
                else:
                    st.error("No valid API keys found. Please check your API keys in Setup.")
                    st.info("👈 **Please select '🔑 Setup' from the navigation menu**")
        
        with col2:
            # Scale selection
            st.markdown("#### Scales")
            scales = ["RWA", "LWA", "MFQ", "NFC"]
            selected_scales = st.multiselect(
                "Select psychological scales",
                scales,
                default=["RWA", "LWA"]
            )
            st.session_state.survey_config['scales'] = selected_scales
        
        # Additional settings
        st.markdown("#### Survey Parameters")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            num_runs = st.number_input(
                "Runs per question",
                min_value=1,
                max_value=10,
                value=1,
                help="Number of times to ask each question"
            )
            st.session_state.survey_config['num_runs'] = num_runs
        
        with col4:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                help="Controls randomness of responses"
            )
            st.session_state.survey_config['temperature'] = temperature
        
        with col5:
            batch_size = st.number_input(
                "Batch size",
                min_value=1,
                max_value=50,
                value=10,
                help="Questions to process simultaneously"
            )
            st.session_state.survey_config['batch_size'] = batch_size
    
    with tabs[1]:
        st.markdown("### Custom Prompt Builder")
        selected_prompts = render_prompt_builder()
        if selected_prompts:
            st.session_state.selected_prompts = selected_prompts
            st.session_state.survey_config['prompts'] = list(selected_prompts.keys())
    
    with tabs[2]:
        st.markdown("### Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Rate Limiting")
            rate_limit = st.checkbox("Enable custom rate limiting", value=False)
            if rate_limit:
                max_rpm = st.number_input("Max requests per minute", value=60)
                st.session_state.survey_config['max_rpm'] = max_rpm
        
        with col2:
            st.markdown("#### Error Handling")
            max_retries = st.number_input("Max retries on error", value=3)
            retry_delay = st.number_input("Retry delay (seconds)", value=2)
            st.session_state.survey_config['max_retries'] = max_retries
            st.session_state.survey_config['retry_delay'] = retry_delay
        
        st.markdown("#### Metadata")
        experiment_name = st.text_input("Experiment name (optional)")
        tags = st.text_input("Tags (comma-separated, optional)")
        researcher_id = st.text_input("Researcher ID (optional)")
        
        if experiment_name or tags or researcher_id:
            st.session_state.survey_config['metadata'] = {
                'experiment_name': experiment_name,
                'tags': tags.split(',') if tags else [],
                'researcher_id': researcher_id
            }
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if validate_configuration():
            st.success("✅ Configuration complete!")
            st.info("👈 **Next step: Select '🚀 Execute' from the navigation menu to run the survey**")
        else:
            st.warning("⚠️ Please complete all configuration fields above")

def show_execute_page():
    """Execution page for running surveys"""
    st.markdown("## 🚀 Survey Execution")
    
    # Check if configuration is ready
    if not st.session_state.get('survey_config', {}).get('models'):
        st.warning("⚠️ No survey configuration found!")
        st.info("👈 **Please select '📝 Configure Survey' from the navigation menu to set up your survey first**")
        return
    
    # Use the integrated executor which handles actual execution
    from frontend.components.survey_executor_integration import render_integrated_executor
    render_integrated_executor()

def show_results_page():
    """Results page for viewing completed surveys"""
    st.markdown("## 📊 Survey Results")
    
    storage = StorageManager()
    recent_runs = storage.get_recent_runs(limit=20)
    
    if not recent_runs:
        st.info("No survey results available yet. Run a survey first!")
        st.info("👈 **To run a survey, select '🚀 Execute' from the navigation menu**")
        return
    
    # Run selector
    st.markdown("### Select Survey Run")
    run_options = {
        f"{run['timestamp'][:19]} - {', '.join(run['summary']['models_used'][:2])}": run['run_id']
        for run in recent_runs
    }
    
    selected_run_label = st.selectbox("Choose a survey run", list(run_options.keys()))
    selected_run_id = run_options[selected_run_label]
    
    # Load and display results
    if selected_run_id:
        survey_data = storage.storage.load_survey_results(selected_run_id)
        
        if survey_data:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Responses", survey_data['summary']['total_responses'])
            with col2:
                st.metric("Success Rate", f"{survey_data['statistics'].get('response_rate', 0):.1f}%")
            with col3:
                total_cost = sum(survey_data.get('costs', {}).values()) if isinstance(survey_data.get('costs', {}), dict) else 0
                st.metric("Total Cost", f"${total_cost:.4f}")
            with col4:
                st.metric("Models Used", len(survey_data['summary']['models_used']))
            
            # Detailed results tabs
            tabs = st.tabs(["📊 Overview", "📈 Statistics", "🔍 Raw Data", "💾 Export"])
            
            with tabs[0]:
                st.markdown("### Survey Configuration")
                st.json(survey_data['configuration'])
                
                if survey_data.get('metadata'):
                    st.markdown("### Metadata")
                    st.json(survey_data['metadata'])
            
            with tabs[1]:
                st.markdown("### Statistical Summary")
                if survey_data.get('statistics'):
                    st.json(survey_data['statistics'])
            
            with tabs[2]:
                st.markdown("### Response Data")
                if survey_data.get('results'):
                    import pandas as pd
                    df = pd.DataFrame(survey_data['results'])
                    st.dataframe(df, use_container_width=True)
            
            with tabs[3]:
                st.markdown("### Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📥 Download as JSON", use_container_width=True):
                        import json
                        json_str = json.dumps(survey_data, indent=2, default=str)
                        st.download_button(
                            "Download JSON",
                            json_str,
                            f"survey_{selected_run_id}.json",
                            "application/json"
                        )
                
                with col2:
                    if st.button("📥 Download as CSV", use_container_width=True):
                        if survey_data.get('results'):
                            import pandas as pd
                            df = pd.DataFrame(survey_data['results'])
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "Download CSV",
                                csv,
                                f"survey_{selected_run_id}.csv",
                                "text/csv"
                            )
                
                with col3:
                    if st.button("📥 Download as Excel", use_container_width=True):
                        if survey_data.get('results'):
                            import pandas as pd
                            import io
                            df = pd.DataFrame(survey_data['results'])
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                df.to_excel(writer, sheet_name='Results', index=False)
                                
                                # Add metadata sheet
                                meta_df = pd.DataFrame([survey_data['summary']])
                                meta_df.to_excel(writer, sheet_name='Summary', index=False)
                            
                            st.download_button(
                                "Download Excel",
                                buffer.getvalue(),
                                f"survey_{selected_run_id}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

def show_explorer_page():
    """Data explorer page for advanced analysis"""
    st.markdown("## 🔍 Data Explorer")
    
    explorer = DataExplorer()
    explorer.render()

def show_documentation_page():
    """Documentation page"""
    st.markdown("## 📚 Documentation")
    
    tabs = st.tabs(["🚀 Getting Started", "📖 User Guide", "🔧 API Reference", "❓ FAQ"])
    
    with tabs[0]:
        st.markdown("""
        ### Getting Started
        
        #### Prerequisites
        - API keys for at least one LLM provider
        - Basic understanding of survey methodology
        - Chrome, Firefox, or Safari browser
        
        #### Quick Start Guide
        
        1. **Obtain API Keys**
           - OpenAI: https://platform.openai.com/api-keys
           - Anthropic: https://console.anthropic.com/settings/keys
           - Other providers: See their respective documentation
        
        2. **Configure Your Survey**
           - Navigate to the Setup page
           - Enter your API keys (they're stored securely in your browser)
           - Select models, scales, and prompts
        
        3. **Run the Survey**
           - Go to the Execute page
           - Review your configuration
           - Click "Start Survey" and monitor progress
        
        4. **Analyze Results**
           - View results immediately after completion
           - Use the Data Explorer for advanced analysis
           - Export data in various formats
        """)
    
    with tabs[1]:
        st.markdown("""
        ### User Guide
        
        #### Understanding Scales
        
        **RWA (Right-Wing Authoritarianism)**
        - Measures authoritarian personality traits
        - Higher scores indicate more authoritarian tendencies
        
        **LWA (Left-Wing Authoritarianism)**
        - Measures left-leaning authoritarian traits
        - Complementary to RWA scale
        
        **MFQ (Moral Foundations Questionnaire)**
        - Assesses moral reasoning across five foundations
        - Care/Harm, Fairness/Cheating, Loyalty/Betrayal, Authority/Subversion, Sanctity/Degradation
        
        **NFC (Need for Closure)**
        - Measures desire for predictability and decisiveness
        - Higher scores indicate greater need for closure
        
        #### Creating Custom Prompts
        
        Use template variables in your prompts:
        - `{QUESTION}` - The survey question
        - `{SCALE}` - The response scale
        - `{OPTIONS}` - Available response options
        - `{SCALE_NAME}` - Name of the psychological scale
        
        Example:
        ```
        As a {persona}, I would answer the following question:
        {QUESTION}
        
        Scale: {SCALE}
        My response: 
        ```
        """)
    
    with tabs[2]:
        st.markdown("""
        ### API Reference
        
        #### Storage Manager
        ```python
        from backend.storage.json_handler import StorageManager
        
        storage = StorageManager()
        
        # Save results
        run_id = storage.save_from_pipeline(results_df, config, metadata)
        
        # Load results
        data = storage.storage.load_survey_results(run_id)
        
        # Search runs
        runs = storage.search_by_model("OpenAI")
        ```
        
        #### Survey Pipeline
        ```python
        from backend.main import SurveyPipeline
        
        pipeline = SurveyPipeline(
            scales_to_run=["RWA", "LWA"],
            prompt_styles_to_run=["minimal"],
            models_to_run=["OpenAI"],
            num_calls_test=1,
            temperature=0.0
        )
        
        results = await pipeline.run_survey()
        ```
        """)
    
    with tabs[3]:
        st.markdown("""
        ### Frequently Asked Questions
        
        **Q: Are my API keys secure?**
        A: Yes, API keys are stored only in your browser session and never sent to our servers.
        
        **Q: How much will a survey cost?**
        A: Costs depend on the models and number of questions. The executor provides estimates before running.
        
        **Q: Can I resume a interrupted survey?**
        A: Currently, surveys must complete in one session. Partial results are saved if you cancel.
        
        **Q: How do I cite this tool?**
        A: [Citation format will be provided]
        
        **Q: Can I add new psychological scales?**
        A: Yes, scales can be added to `backend/config/scales.py` following the existing format.
        """)

def validate_configuration():
    """Validate survey configuration"""
    config = st.session_state.survey_config
    
    if not config.get('models'):
        st.error("Please select at least one model")
        return False
    
    if not config.get('scales'):
        st.error("Please select at least one scale")
        return False
    
    if not st.session_state.selected_prompts:
        st.error("Please select or create at least one prompt")
        return False
    
    return True

if __name__ == "__main__":
    main()