"""Integration module for Survey Executor with the main pipeline.

This module provides integration functions to connect the survey executor
component with the existing backend pipeline and dashboard.
"""

import streamlit as st
import asyncio
from typing import Dict, List, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from backend.config.models import MODEL_CONFIG
from backend.config.prompts import prompt_templates
from backend.config.scales import all_scales, all_questions
from main import SurveyPipeline
from frontend.components.survey_executor import SurveyExecutor, render_survey_executor


def prepare_survey_config(
    selected_models: List[str],
    selected_scales: List[str],
    selected_prompts: List[str],
    num_runs: int = 1,
    temperature: float = 0.0
) -> Dict:
    """Prepare survey configuration for the executor.
    
    Args:
        selected_models: List of model names to test
        selected_scales: List of scale names to run
        selected_prompts: List of prompt styles to use
        num_runs: Number of runs per question
        temperature: Temperature for model responses
        
    Returns:
        Configuration dictionary for the survey executor
    """
    # Filter questions based on selected scales
    questions = []
    for scale_list in all_scales:
        for q in scale_list:
            if q["scale_name"] in selected_scales:
                questions.append({
                    'id': q['id'],
                    'text': q['text'],
                    'scale_name': q['scale_name'],
                    'scale_range': q['scale_range'],
                    'reverse_score': q.get('reverse_score', False)
                })
    
    config = {
        'models': selected_models,
        'scales': selected_scales,
        'prompts': selected_prompts,
        'questions': questions,
        'runs': num_runs,
        'temperature': temperature
    }
    
    return config


def create_pipeline_instance(config: Dict) -> SurveyPipeline:
    """Create a SurveyPipeline instance from configuration.
    
    Args:
        config: Survey configuration dictionary
        
    Returns:
        Configured SurveyPipeline instance
    """
    pipeline = SurveyPipeline(
        scales_to_run=config['scales'],
        prompt_styles_to_run=config['prompts'],
        models_to_run=config['models'],
        num_calls_test=config['runs'],
        temperature=config['temperature']
    )
    
    return pipeline


def render_integrated_executor():
    """Render the survey executor with full integration to the dashboard."""
    
    st.header("🚀 Survey Execution Center")
    st.markdown("Configure and execute your LLM survey with real-time progress tracking.")
    
    # Check if we have a configuration from the main dashboard
    if 'survey_config' not in st.session_state:
        # Provide configuration interface
        with st.container():
            st.subheader("📝 Quick Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Model selection
                available_models = [m for m in MODEL_CONFIG.keys() if MODEL_CONFIG[m].get('api_key')]
                if not available_models:
                    st.error("⚠️ No models with API keys configured. Please set up API keys first.")
                    return
                
                selected_models = st.multiselect(
                    "Select Models",
                    available_models,
                    default=available_models[:2] if len(available_models) >= 2 else available_models,
                    help="Choose which LLM models to survey"
                )
                
                # Scale selection
                scale_names = list(set(q["scale_name"] for scale_list in all_scales for q in scale_list))
                selected_scales = st.multiselect(
                    "Select Scales",
                    scale_names,
                    default=scale_names[:2] if len(scale_names) >= 2 else scale_names,
                    help="Choose psychological scales to administer"
                )
            
            with col2:
                # Prompt style selection
                selected_prompts = st.multiselect(
                    "Select Prompt Styles",
                    list(prompt_templates.keys()),
                    default=list(prompt_templates.keys())[:2],
                    help="Choose prompting personas"
                )
                
                # Parameters
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    num_runs = st.number_input(
                        "Runs per Question",
                        min_value=1,
                        max_value=10,
                        value=1,
                        help="Number of times to ask each question"
                    )
                
                with col2_2:
                    temperature = st.slider(
                        "Temperature",
                        min_value=0.0,
                        max_value=2.0,
                        value=0.0,
                        step=0.1,
                        help="Response randomness (0=deterministic)"
                    )
            
            # Prepare configuration button
            if st.button("✅ Prepare Survey", type="primary", use_container_width=True):
                if not selected_models:
                    st.error("Please select at least one model.")
                elif not selected_scales:
                    st.error("Please select at least one scale.")
                elif not selected_prompts:
                    st.error("Please select at least one prompt style.")
                else:
                    # Prepare configuration
                    config = prepare_survey_config(
                        selected_models,
                        selected_scales,
                        selected_prompts,
                        num_runs,
                        temperature
                    )
                    
                    # Store configuration and pipeline
                    st.session_state['survey_config'] = config
                    st.session_state['pipeline'] = create_pipeline_instance(config)
                    st.rerun()
    
    # If configuration exists, show the executor
    if 'survey_config' in st.session_state:
        st.markdown("---")
        
        # Get the configuration
        config = st.session_state['survey_config']
        
        # Ensure prompts are properly set from selected_prompts
        if 'selected_prompts' in st.session_state and st.session_state.selected_prompts:
            config['prompts'] = list(st.session_state.selected_prompts.keys())
        elif 'prompts' not in config or not config['prompts']:
            config['prompts'] = ['minimal']
        
        # Ensure pipeline instance exists
        if 'pipeline' not in st.session_state:
            st.session_state['pipeline'] = create_pipeline_instance(config)
        
        # Add configuration summary
        with st.expander("📋 Current Configuration", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Models:**")
                for model in config['models']:
                    st.write(f"• {model}")
            with col2:
                st.write("**Scales:**")
                for scale in config['scales']:
                    st.write(f"• {scale}")
            with col3:
                st.write("**Prompts:**")
                for prompt in config['prompts']:
                    st.write(f"• {prompt}")
            
            if st.button("🔄 Reconfigure", use_container_width=True):
                del st.session_state['survey_config']
                if 'pipeline' in st.session_state:
                    del st.session_state['pipeline']
                st.rerun()
        
        # Initialize executor if needed
        if 'executor' not in st.session_state:
            st.session_state['executor'] = SurveyExecutor()
        
        executor = st.session_state['executor']
        
        # Handle execution based on status
        status = st.session_state.execution_state['status']
        
        if status == 'idle':
            executor.render_pre_execution(config)
        
        elif status == 'running':
            # Check if we need to start the execution
            if 'execution_started' not in st.session_state:
                st.session_state.execution_started = True
                
                # Run the survey in a separate thread to not block UI
                pipeline = st.session_state['pipeline']
                
                # Build tasks first to verify we have work to do
                tasks = pipeline.build_tasks()
                st.session_state.execution_state['total_tasks'] = len(tasks)
                
                if len(tasks) == 0:
                    st.error("No tasks to execute! Check your configuration.")
                    st.write("Debug Info:")
                    st.write(f"Models: {pipeline.models_to_run}")
                    st.write(f"Scales: {pipeline.scales_to_run}")
                    st.write(f"Prompts: {pipeline.prompt_styles_to_run}")
                    st.write(f"Runs: {pipeline.num_calls_test}")
                    st.session_state.execution_state['status'] = 'error'
                    st.rerun()
                else:
                    # Start async execution
                    with st.spinner(f"Initializing {len(tasks)} tasks..."):
                        try:
                            # Run the async execution
                            results = asyncio.run(executor.execute_survey_async(pipeline, config))
                            st.session_state.execution_state['results'] = results
                            st.session_state.execution_state['status'] = 'completed'
                            del st.session_state.execution_started
                            st.rerun()
                        except Exception as e:
                            st.error(f"Execution failed: {str(e)}")
                            st.session_state.execution_state['status'] = 'error'
                            if 'execution_started' in st.session_state:
                                del st.session_state.execution_started
            else:
                # Show progress while execution is running
                executor.render_execution_progress()
        
        elif status == 'paused':
            executor.render_execution_progress()
        
        elif status == 'completed':
            executor.render_post_execution()
        
        elif status == 'error':
            st.error("❌ Survey execution encountered an error.")
            errors = st.session_state.execution_state.get('errors', [])
            if errors:
                with st.expander("View Error Details", expanded=True):
                    for error in errors[-5:]:
                        st.error(f"{error['timestamp']} - {error['model']}: {error['message']}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Retry", type="primary", use_container_width=True):
                    st.session_state.execution_state['status'] = 'idle'
                    st.session_state.execution_state['errors'] = []
                    st.rerun()
            with col2:
                if st.button("❌ Reset", type="secondary", use_container_width=True):
                    if 'execution_state' in st.session_state:
                        del st.session_state['execution_state']
                    st.rerun()


def get_execution_results() -> Optional[Dict]:
    """Get the results from the last survey execution.
    
    Returns:
        Dictionary containing results and metadata, or None if no results
    """
    if 'execution_state' not in st.session_state:
        return None
    
    state = st.session_state.execution_state
    
    if state.get('status') != 'completed' or state.get('results') is None:
        return None
    
    return {
        'results': state['results'],
        'metadata': {
            'total_tasks': state.get('total_tasks', 0),
            'completed_tasks': state.get('completed_tasks', 0),
            'failed_tasks': state.get('failed_tasks', 0),
            'total_cost': state.get('cost_accumulator', 0.0),
            'token_usage': dict(state.get('token_usage', {})),
            'start_time': state.get('start_time'),
            'end_time': state.get('end_time'),
            'errors': state.get('errors', [])
        }
    }


def clear_execution_state():
    """Clear the execution state and results."""
    if 'execution_state' in st.session_state:
        del st.session_state['execution_state']
    if 'executor' in st.session_state:
        del st.session_state['executor']
    if 'survey_config' in st.session_state:
        del st.session_state['survey_config']
    if 'pipeline' in st.session_state:
        del st.session_state['pipeline']


# Example standalone app for testing
if __name__ == "__main__":
    st.set_page_config(
        page_title="Survey Executor Integration Test",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Integrated Survey Executor")
    
    # Add a sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Select Page", ["Executor", "Results"])
    
    if page == "Executor":
        render_integrated_executor()
    
    elif page == "Results":
        st.header("📊 Execution Results")
        
        results_data = get_execution_results()
        
        if results_data:
            st.success("✅ Results available from last execution")
            
            # Show metadata
            metadata = results_data['metadata']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Completed Tasks", metadata['completed_tasks'])
            with col2:
                st.metric("Failed Tasks", metadata['failed_tasks'])
            with col3:
                st.metric("Total Cost", f"${metadata['total_cost']:.2f}")
            with col4:
                duration = metadata['end_time'] - metadata['start_time']
                st.metric("Duration", f"{duration.seconds // 60}m")
            
            # Show results
            st.subheader("Survey Responses")
            st.dataframe(results_data['results'].head(20), use_container_width=True)
            
            # Clear button
            if st.button("🗑️ Clear All Results", type="secondary"):
                clear_execution_state()
                st.rerun()
        else:
            st.info("No results available. Please run a survey first.")