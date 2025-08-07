"""
Example integration of the Prompt Builder component into the main dashboard
This shows how to use the prompt builder in your survey pipeline
"""

import streamlit as st
from prompt_builder import render_prompt_builder
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.prompts import prompt_templates


def integrate_prompt_builder_with_survey():
    """
    Example of how to integrate the prompt builder with your survey system
    """
    st.title("Survey Configuration with Custom Prompts")
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["Prompt Builder", "Survey Configuration", "Run Survey"])
    
    with tab1:
        # Get selected prompts from the builder
        selected_prompts = render_prompt_builder()
        
    with tab2:
        st.header("Survey Configuration")
        
        if selected_prompts:
            st.success(f"You have selected {len(selected_prompts)} prompts")
            
            # Show selected prompts
            st.subheader("Selected Prompts:")
            for key, prompt_data in selected_prompts.items():
                with st.expander(prompt_data["name"]):
                    st.text(prompt_data.get("description", ""))
                    st.code(prompt_data["template"], language="text")
            
            # Other survey configuration options
            st.subheader("Additional Settings")
            
            # Scale selection
            scales = st.multiselect(
                "Select Scales",
                ["RWA", "LWA", "MFQ", "NFC"],
                help="Choose which psychological scales to administer"
            )
            
            # Model selection
            models = st.multiselect(
                "Select Models",
                ["OpenAI", "Claude", "Llama", "DeepSeek", "Grok"],
                help="Choose which LLM models to test"
            )
            
            # Number of runs
            num_runs = st.number_input(
                "Number of Runs",
                min_value=1,
                max_value=10,
                value=3,
                help="How many times to run each prompt-model combination"
            )
            
            # Store configuration
            if st.button("Save Configuration", type="primary"):
                st.session_state.survey_config = {
                    "prompts": selected_prompts,
                    "scales": scales,
                    "models": models,
                    "runs": num_runs
                }
                st.success("Configuration saved! Go to 'Run Survey' tab to execute.")
        else:
            st.info("Please select prompts in the Prompt Builder tab first")
    
    with tab3:
        st.header("Run Survey")
        
        if "survey_config" in st.session_state:
            config = st.session_state.survey_config
            
            # Show configuration summary
            st.subheader("Survey Configuration Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Prompts", len(config["prompts"]))
            with col2:
                st.metric("Scales", len(config["scales"]))
            with col3:
                st.metric("Models", len(config["models"]))
            with col4:
                st.metric("Runs", config["runs"])
            
            # Calculate total tasks
            total_tasks = (
                len(config["prompts"]) * 
                len(config["scales"]) * 
                len(config["models"]) * 
                config["runs"]
            )
            
            st.info(f"This will create {total_tasks} survey tasks")
            
            # Run button
            if st.button("Run Survey", type="primary", use_container_width=True):
                # Convert selected prompts to format expected by pipeline
                pipeline_prompts = {}
                for key, prompt_data in config["prompts"].items():
                    # Use the template directly
                    pipeline_prompts[key] = prompt_data["template"]
                
                # Here you would integrate with your actual survey pipeline
                st.info("Survey execution would start here...")
                
                # Example of how to use the prompts
                with st.spinner("Running surveys..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate running surveys
                    for i in range(total_tasks):
                        progress_bar.progress((i + 1) / total_tasks)
                        status_text.text(f"Processing task {i + 1} of {total_tasks}")
                    
                    st.success("Survey completed successfully!")
                    
                    # Show sample results
                    st.subheader("Sample Results")
                    st.json({
                        "prompts_used": list(config["prompts"].keys()),
                        "scales_tested": config["scales"],
                        "models_tested": config["models"],
                        "total_responses": total_tasks,
                        "status": "completed"
                    })
        else:
            st.info("Please configure your survey in the 'Survey Configuration' tab first")


def convert_prompts_for_pipeline(selected_prompts):
    """
    Convert the prompt builder format to the format expected by the existing pipeline
    
    Args:
        selected_prompts: Dictionary from prompt builder
        
    Returns:
        Dictionary in pipeline format
    """
    converted = {}
    for key, prompt_data in selected_prompts.items():
        # Extract just the template string
        converted[key] = prompt_data["template"]
    
    return converted


def merge_with_existing_prompts(custom_prompts):
    """
    Merge custom prompts with existing prompts from config
    
    Args:
        custom_prompts: Dictionary of custom prompts
        
    Returns:
        Merged dictionary of all prompts
    """
    # Start with existing prompts
    all_prompts = prompt_templates.copy()
    
    # Add custom prompts
    for key, template in custom_prompts.items():
        # Ensure unique keys
        unique_key = key
        counter = 1
        while unique_key in all_prompts:
            unique_key = f"{key}_{counter}"
            counter += 1
        
        all_prompts[unique_key] = template
    
    return all_prompts


if __name__ == "__main__":
    st.set_page_config(
        page_title="Survey Pipeline with Custom Prompts",
        page_icon="📊",
        layout="wide"
    )
    
    integrate_prompt_builder_with_survey()