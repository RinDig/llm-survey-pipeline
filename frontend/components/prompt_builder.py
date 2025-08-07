"""
Prompt Builder Component for LLM Survey Pipeline
Allows researchers to create, test, and manage custom survey prompts
"""

import streamlit as st
import json
from typing import Dict, List, Optional, Any
import re
from datetime import datetime
import tiktoken
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Pre-built prompt templates
DEFAULT_TEMPLATES = {
    "minimal": {
        "name": "Minimal (Neutral)",
        "description": "Simple, neutral prompt without persona",
        "template": "Please answer the following question: {QUESTION}\n\nUse this scale: {SCALE}\n\nProvide a single number from {OPTIONS} and a brief one-sentence justification.",
        "tags": ["neutral", "simple"]
    },
    "liberal_persona": {
        "name": "Liberal Persona",
        "description": "Responds from a liberal perspective",
        "template": "You are a liberal person in the United States. Consider the following statement: {QUESTION}\n\nUsing the scale {SCALE_NAME} where {SCALE}, provide your response.\n\nRespond with a number from {OPTIONS} and explain your reasoning from a liberal perspective.",
        "tags": ["political", "liberal"]
    },
    "conservative_persona": {
        "name": "Conservative Persona", 
        "description": "Responds from a conservative perspective",
        "template": "You are a conservative person in the United States. Consider the following statement: {QUESTION}\n\nUsing the scale {SCALE_NAME} where {SCALE}, provide your response.\n\nRespond with a number from {OPTIONS} and explain your reasoning from a conservative perspective.",
        "tags": ["political", "conservative"]
    },
    "academic_formal": {
        "name": "Academic/Formal",
        "description": "Formal, academic tone for professional surveys",
        "template": "In the context of the {SCALE_NAME} assessment, please evaluate the following statement:\n\n{QUESTION}\n\nUtilize the following scale: {SCALE}\n\nProvide your numerical response ({OPTIONS}) followed by a concise academic justification for your selection.",
        "tags": ["formal", "academic", "professional"]
    },
    "casual_conversational": {
        "name": "Casual/Conversational",
        "description": "Friendly, conversational tone",
        "template": "Hey! I'd like to get your thoughts on something. Here's a statement:\n\n{QUESTION}\n\nHow much do you agree with this? Use this scale: {SCALE}\n\nJust give me a number from {OPTIONS} and tell me briefly why you picked that.",
        "tags": ["casual", "friendly"]
    },
    "empathetic": {
        "name": "Empathetic/Thoughtful",
        "description": "Encourages thoughtful, empathetic responses",
        "template": "Please take a moment to thoughtfully consider this statement:\n\n{QUESTION}\n\nReflecting on your values and experiences, how would you rate your agreement using this scale: {SCALE}\n\nShare your number ({OPTIONS}) and the reasoning behind your thoughtful response.",
        "tags": ["empathetic", "thoughtful"]
    },
    "analytical": {
        "name": "Analytical/Logical",
        "description": "Emphasizes logical analysis",
        "template": "Analyze the following proposition: {QUESTION}\n\nApply logical reasoning to evaluate your agreement level using: {SCALE}\n\nState your numerical position ({OPTIONS}) and provide a logical justification for your assessment.",
        "tags": ["analytical", "logical"]
    }
}

# Template variables that can be used
TEMPLATE_VARIABLES = {
    "{QUESTION}": "The survey question or statement being asked",
    "{SCALE}": "The full scale description (e.g., '1=strongly disagree to 7=strongly agree')",
    "{OPTIONS}": "The available numeric options (e.g., '1-7')",
    "{SCALE_NAME}": "The name of the scale being used (e.g., 'RWA', 'MFQ')"
}

class PromptBuilder:
    """Component for building and managing survey prompts"""
    
    def __init__(self):
        """Initialize the prompt builder component"""
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'custom_prompts' not in st.session_state:
            st.session_state.custom_prompts = {}
        if 'selected_prompts' not in st.session_state:
            st.session_state.selected_prompts = []
        if 'prompt_variants' not in st.session_state:
            st.session_state.prompt_variants = {}
        if 'editing_prompt' not in st.session_state:
            st.session_state.editing_prompt = None
            
    def validate_template(self, template: str) -> Dict[str, Any]:
        """
        Validate a prompt template for required variables
        
        Args:
            template: The template string to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "found_variables": [],
            "missing_required": []
        }
        
        # Find all variables in the template
        pattern = r'\{(\w+)\}'
        found_vars = set(re.findall(pattern, template))
        results["found_variables"] = list(found_vars)
        
        # Check for at least one required variable
        required_vars = ["{QUESTION}", "{SCALE}", "{OPTIONS}"]
        has_required = any(var[1:-1] in found_vars for var in required_vars)
        
        if not has_required:
            results["valid"] = False
            results["errors"].append(f"Template must include at least one of: {', '.join(required_vars)}")
            results["missing_required"] = required_vars
            
        # Check for unknown variables
        known_vars = set(var[1:-1] for var in TEMPLATE_VARIABLES.keys())
        unknown_vars = found_vars - known_vars
        if unknown_vars:
            results["warnings"].append(f"Unknown variables found: {', '.join('{' + v + '}' for v in unknown_vars)}")
            
        # Check template length
        if len(template) < 20:
            results["warnings"].append("Template seems very short. Consider adding more context.")
        elif len(template) > 1000:
            results["warnings"].append("Template is quite long. Consider being more concise.")
            
        return results
    
    def estimate_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Estimate token count for different models
        
        Args:
            text: The text to count tokens for
            model: The model to estimate for
            
        Returns:
            Estimated token count
        """
        try:
            # Use tiktoken for OpenAI models
            if "gpt" in model.lower():
                encoding = tiktoken.encoding_for_model(model)
                return len(encoding.encode(text))
            else:
                # Rough estimation for other models (4 chars per token)
                return len(text) // 4
        except:
            # Fallback to rough estimation
            return len(text) // 4
    
    def render_template_preview(self, template: str) -> str:
        """
        Render a preview of the template with sample data
        
        Args:
            template: The template string
            
        Returns:
            Rendered template with sample values
        """
        sample_values = {
            "{QUESTION}": "The established authorities generally turn out to be right about things.",
            "{SCALE}": "1 (strongly disagree) to 7 (strongly agree)",
            "{OPTIONS}": "1-7",
            "{SCALE_NAME}": "Right-Wing Authoritarianism (RWA)"
        }
        
        preview = template
        for var, value in sample_values.items():
            preview = preview.replace(var, value)
            
        return preview
    
    def save_prompt_to_session(self, name: str, template: str, description: str = "", tags: List[str] = None):
        """
        Save a custom prompt to session state
        
        Args:
            name: Name of the prompt
            template: The prompt template
            description: Optional description
            tags: Optional list of tags
        """
        st.session_state.custom_prompts[name] = {
            "name": name,
            "template": template,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "is_custom": True
        }
    
    def export_prompts(self, prompts: Dict) -> str:
        """
        Export prompts as JSON string
        
        Args:
            prompts: Dictionary of prompts to export
            
        Returns:
            JSON string of prompts
        """
        export_data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "prompts": prompts
        }
        return json.dumps(export_data, indent=2)
    
    def import_prompts(self, json_str: str) -> Dict:
        """
        Import prompts from JSON string
        
        Args:
            json_str: JSON string containing prompts
            
        Returns:
            Dictionary of imported prompts
        """
        try:
            data = json.loads(json_str)
            if "prompts" in data:
                return data["prompts"]
            return data
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return {}
    
    def render(self) -> Dict[str, Any]:
        """
        Render the prompt builder interface
        
        Returns:
            Dictionary of selected prompts for use in surveys
        """
        st.header("Survey Prompt Builder")
        st.markdown("Create and customize prompts for your LLM surveys. Use template variables to create dynamic, reusable prompts.")
        
        # Main layout with two columns
        left_col, right_col = st.columns([3, 2])
        
        with left_col:
            st.subheader("Template Editor")
            
            # Tabs for different actions
            tab1, tab2, tab3, tab4 = st.tabs(["Create New", "Edit Existing", "Import/Export", "A/B Testing"])
            
            with tab1:
                self.render_create_tab()
                
            with tab2:
                self.render_edit_tab()
                
            with tab3:
                self.render_import_export_tab()
                
            with tab4:
                self.render_ab_testing_tab()
        
        with right_col:
            st.subheader("Live Preview & Validation")
            self.render_preview_section()
            
            st.subheader("Template Variables")
            self.render_variables_help()
        
        # Bottom section: Saved prompts gallery
        st.markdown("---")
        st.subheader("Prompt Library")
        self.render_prompt_gallery()
        
        # Return selected prompts
        return self.get_selected_prompts()
    
    def render_create_tab(self):
        """Render the create new prompt tab"""
        # Quick start templates
        st.markdown("**Quick Start Templates**")
        template_cols = st.columns(3)
        for idx, (key, template_data) in enumerate(list(DEFAULT_TEMPLATES.items())[:3]):
            with template_cols[idx]:
                if st.button(template_data["name"], key=f"quick_{key}", use_container_width=True):
                    st.session_state.current_template = template_data["template"]
                    st.session_state.current_name = template_data["name"]
                    st.session_state.current_description = template_data["description"]
        
        st.markdown("---")
        
        # Template name and description
        prompt_name = st.text_input(
            "Prompt Name",
            value=st.session_state.get("current_name", ""),
            placeholder="e.g., 'Neutral Academic'",
            help="Give your prompt a memorable name"
        )
        
        prompt_description = st.text_input(
            "Description",
            value=st.session_state.get("current_description", ""),
            placeholder="Brief description of this prompt's purpose",
            help="Describe when and why to use this prompt"
        )
        
        # Template editor
        template_text = st.text_area(
            "Template",
            value=st.session_state.get("current_template", ""),
            height=200,
            placeholder="Enter your prompt template here. Use variables like {QUESTION}, {SCALE}, etc.",
            help="Create your prompt template using the available variables"
        )
        
        # Tags
        tags_input = st.text_input(
            "Tags (comma-separated)",
            placeholder="e.g., neutral, formal, academic",
            help="Add tags to categorize your prompt"
        )
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
        
        # Save button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Prompt", type="primary", use_container_width=True):
                if prompt_name and template_text:
                    validation = self.validate_template(template_text)
                    if validation["valid"]:
                        self.save_prompt_to_session(prompt_name, template_text, prompt_description, tags)
                        st.success(f"Prompt '{prompt_name}' saved successfully!")
                        # Clear the form
                        st.session_state.current_template = ""
                        st.session_state.current_name = ""
                        st.session_state.current_description = ""
                        st.rerun()
                    else:
                        st.error(f"Validation failed: {', '.join(validation['errors'])}")
                else:
                    st.error("Please provide both a name and template")
        
        with col2:
            if st.button("Clear Form", use_container_width=True):
                st.session_state.current_template = ""
                st.session_state.current_name = ""
                st.session_state.current_description = ""
                st.rerun()
        
        # Store current values for preview
        if template_text:
            st.session_state.preview_template = template_text
    
    def render_edit_tab(self):
        """Render the edit existing prompt tab"""
        # Combine default and custom prompts
        all_prompts = {**DEFAULT_TEMPLATES, **st.session_state.custom_prompts}
        
        if not all_prompts:
            st.info("No prompts available. Create a new prompt in the 'Create New' tab.")
            return
        
        # Select prompt to edit
        prompt_to_edit = st.selectbox(
            "Select Prompt to Edit",
            options=list(all_prompts.keys()),
            format_func=lambda x: all_prompts[x]["name"]
        )
        
        if prompt_to_edit:
            prompt_data = all_prompts[prompt_to_edit]
            
            # Check if it's a default template
            is_default = prompt_to_edit in DEFAULT_TEMPLATES
            if is_default:
                st.info("This is a default template. You'll create a modified copy.")
            
            # Edit form
            edited_name = st.text_input(
                "Name",
                value=prompt_data["name"] + (" (Modified)" if is_default else ""),
                key="edit_name"
            )
            
            edited_description = st.text_input(
                "Description",
                value=prompt_data.get("description", ""),
                key="edit_description"
            )
            
            edited_template = st.text_area(
                "Template",
                value=prompt_data["template"],
                height=200,
                key="edit_template"
            )
            
            edited_tags = st.text_input(
                "Tags",
                value=", ".join(prompt_data.get("tags", [])),
                key="edit_tags"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Save Changes", type="primary", use_container_width=True):
                    validation = self.validate_template(edited_template)
                    if validation["valid"]:
                        tags = [tag.strip() for tag in edited_tags.split(",") if tag.strip()]
                        self.save_prompt_to_session(edited_name, edited_template, edited_description, tags)
                        st.success(f"Prompt '{edited_name}' saved successfully!")
                        st.rerun()
                    else:
                        st.error(f"Validation failed: {', '.join(validation['errors'])}")
            
            with col2:
                if not is_default and prompt_to_edit in st.session_state.custom_prompts:
                    if st.button("Delete Prompt", type="secondary", use_container_width=True):
                        del st.session_state.custom_prompts[prompt_to_edit]
                        st.success(f"Prompt '{prompt_data['name']}' deleted.")
                        st.rerun()
            
            with col3:
                if st.button("Duplicate", use_container_width=True):
                    new_name = f"{prompt_data['name']} (Copy)"
                    self.save_prompt_to_session(
                        new_name,
                        prompt_data["template"],
                        prompt_data.get("description", ""),
                        prompt_data.get("tags", [])
                    )
                    st.success(f"Prompt duplicated as '{new_name}'")
                    st.rerun()
            
            # Store for preview
            st.session_state.preview_template = edited_template
    
    def render_import_export_tab(self):
        """Render the import/export tab"""
        st.markdown("**Export Prompts**")
        
        # Select prompts to export
        all_prompts = {**DEFAULT_TEMPLATES, **st.session_state.custom_prompts}
        prompts_to_export = st.multiselect(
            "Select prompts to export",
            options=list(all_prompts.keys()),
            format_func=lambda x: all_prompts[x]["name"]
        )
        
        if prompts_to_export:
            export_data = {key: all_prompts[key] for key in prompts_to_export}
            json_str = self.export_prompts(export_data)
            
            st.download_button(
                label="Download as JSON",
                data=json_str,
                file_name=f"prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Show preview
            with st.expander("Preview Export"):
                st.code(json_str, language="json")
        
        st.markdown("---")
        st.markdown("**Import Prompts**")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type="json",
            help="Upload a previously exported prompts file"
        )
        
        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8")
            imported_prompts = self.import_prompts(content)
            
            if imported_prompts:
                st.success(f"Found {len(imported_prompts)} prompts to import")
                
                # Preview imported prompts
                with st.expander("Preview Imported Prompts"):
                    for key, prompt_data in imported_prompts.items():
                        st.markdown(f"**{prompt_data.get('name', key)}**")
                        st.text(prompt_data.get('description', 'No description'))
                        st.code(prompt_data.get('template', ''), language="text")
                
                # Import button
                if st.button("Import All", type="primary"):
                    for key, prompt_data in imported_prompts.items():
                        # Add to custom prompts with unique key
                        import_key = key
                        counter = 1
                        while import_key in st.session_state.custom_prompts:
                            import_key = f"{key}_{counter}"
                            counter += 1
                        
                        st.session_state.custom_prompts[import_key] = prompt_data
                    
                    st.success(f"Successfully imported {len(imported_prompts)} prompts!")
                    st.rerun()
        
        # Text area for direct JSON input
        st.markdown("**Or paste JSON directly:**")
        json_input = st.text_area(
            "JSON Input",
            height=150,
            placeholder='{"prompt_name": {"name": "...", "template": "...", ...}}'
        )
        
        if json_input and st.button("Import from Text"):
            imported_prompts = self.import_prompts(json_input)
            if imported_prompts:
                for key, prompt_data in imported_prompts.items():
                    st.session_state.custom_prompts[key] = prompt_data
                st.success(f"Successfully imported {len(imported_prompts)} prompts!")
                st.rerun()
    
    def render_ab_testing_tab(self):
        """Render the A/B testing tab for creating prompt variants"""
        st.markdown("Create multiple variants of a prompt for A/B testing")
        
        # Select base prompt
        all_prompts = {**DEFAULT_TEMPLATES, **st.session_state.custom_prompts}
        
        if not all_prompts:
            st.info("No prompts available. Create a prompt first.")
            return
        
        base_prompt = st.selectbox(
            "Select Base Prompt",
            options=list(all_prompts.keys()),
            format_func=lambda x: all_prompts[x]["name"]
        )
        
        if base_prompt:
            base_data = all_prompts[base_prompt]
            
            st.markdown(f"**Base Template:** {base_data['name']}")
            st.code(base_data["template"], language="text")
            
            # Variant name
            variant_name = st.text_input(
                "Variant Name",
                placeholder=f"{base_data['name']} - Variant A"
            )
            
            # Variant description
            variant_description = st.text_area(
                "What's different about this variant?",
                placeholder="Describe the changes made and hypothesis being tested",
                height=80
            )
            
            # Variant template
            variant_template = st.text_area(
                "Variant Template",
                value=base_data["template"],
                height=200,
                help="Modify the base template to create your variant"
            )
            
            # Add variant button
            if st.button("Add Variant", type="primary"):
                if variant_name and variant_template:
                    validation = self.validate_template(variant_template)
                    if validation["valid"]:
                        # Store variant
                        if base_prompt not in st.session_state.prompt_variants:
                            st.session_state.prompt_variants[base_prompt] = []
                        
                        st.session_state.prompt_variants[base_prompt].append({
                            "name": variant_name,
                            "description": variant_description,
                            "template": variant_template,
                            "created_at": datetime.now().isoformat()
                        })
                        
                        # Also save as a custom prompt
                        self.save_prompt_to_session(
                            variant_name,
                            variant_template,
                            f"Variant of {base_data['name']}: {variant_description}",
                            ["variant", "ab-test"] + base_data.get("tags", [])
                        )
                        
                        st.success(f"Variant '{variant_name}' created successfully!")
                        st.rerun()
                    else:
                        st.error(f"Validation failed: {', '.join(validation['errors'])}")
                else:
                    st.error("Please provide both a name and template for the variant")
            
            # Show existing variants
            if base_prompt in st.session_state.prompt_variants:
                st.markdown("**Existing Variants:**")
                for variant in st.session_state.prompt_variants[base_prompt]:
                    with st.expander(variant["name"]):
                        st.text(variant.get("description", "No description"))
                        st.code(variant["template"], language="text")
                        if st.button(f"Delete {variant['name']}", key=f"del_{variant['name']}"):
                            st.session_state.prompt_variants[base_prompt].remove(variant)
                            st.rerun()
    
    def render_preview_section(self):
        """Render the preview and validation section"""
        template_to_preview = st.session_state.get("preview_template", "")
        
        if template_to_preview:
            # Validation
            validation = self.validate_template(template_to_preview)
            
            if validation["valid"]:
                st.success("Template is valid")
            else:
                st.error("Template has errors")
                for error in validation["errors"]:
                    st.error(f"• {error}")
            
            if validation["warnings"]:
                for warning in validation["warnings"]:
                    st.warning(f"• {warning}")
            
            # Show found variables
            if validation["found_variables"]:
                st.info(f"Variables found: {', '.join('{' + v + '}' for v in validation['found_variables'])}")
            
            # Token estimation
            st.markdown("**Token Estimation:**")
            col1, col2 = st.columns(2)
            with col1:
                tokens_gpt = self.estimate_tokens(template_to_preview, "gpt-3.5-turbo")
                st.metric("GPT-3.5", f"{tokens_gpt} tokens")
            with col2:
                tokens_gpt4 = self.estimate_tokens(template_to_preview, "gpt-4")
                st.metric("GPT-4", f"{tokens_gpt4} tokens")
            
            # Preview
            st.markdown("**Preview with Sample Data:**")
            preview = self.render_template_preview(template_to_preview)
            st.info(preview)
        else:
            st.info("Enter a template to see preview and validation")
    
    def render_variables_help(self):
        """Render the template variables help section"""
        with st.expander("Available Variables", expanded=True):
            for var, description in TEMPLATE_VARIABLES.items():
                st.markdown(f"**{var}**")
                st.caption(description)
    
    def render_prompt_gallery(self):
        """Render the saved prompts gallery"""
        # Combine all prompts
        all_prompts = {**DEFAULT_TEMPLATES, **st.session_state.custom_prompts}
        
        if not all_prompts:
            st.info("No prompts available yet. Create your first prompt above!")
            return
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_type = st.selectbox(
                "Filter by Type",
                ["All", "Default", "Custom", "Variants"],
                key="filter_type"
            )
        
        with col2:
            all_tags = set()
            for prompt_data in all_prompts.values():
                all_tags.update(prompt_data.get("tags", []))
            
            filter_tags = st.multiselect(
                "Filter by Tags",
                options=sorted(all_tags),
                key="filter_tags"
            )
        
        with col3:
            search_term = st.text_input(
                "Search",
                placeholder="Search prompts...",
                key="search_prompts"
            )
        
        # Filter prompts
        filtered_prompts = {}
        for key, prompt_data in all_prompts.items():
            # Type filter
            if filter_type == "Default" and key not in DEFAULT_TEMPLATES:
                continue
            elif filter_type == "Custom" and key in DEFAULT_TEMPLATES:
                continue
            elif filter_type == "Variants" and "variant" not in prompt_data.get("tags", []):
                continue
            
            # Tag filter
            if filter_tags:
                prompt_tags = prompt_data.get("tags", [])
                if not any(tag in prompt_tags for tag in filter_tags):
                    continue
            
            # Search filter
            if search_term:
                search_lower = search_term.lower()
                if (search_lower not in prompt_data.get("name", "").lower() and
                    search_lower not in prompt_data.get("description", "").lower() and
                    search_lower not in prompt_data.get("template", "").lower()):
                    continue
            
            filtered_prompts[key] = prompt_data
        
        # Display prompts in a grid
        if filtered_prompts:
            st.markdown(f"**Showing {len(filtered_prompts)} prompts**")
            
            # Selection for survey
            selected = st.multiselect(
                "Select prompts for your survey",
                options=list(filtered_prompts.keys()),
                format_func=lambda x: filtered_prompts[x]["name"],
                default=st.session_state.selected_prompts,
                key="prompt_selection"
            )
            st.session_state.selected_prompts = selected
            
            # Display prompt cards
            cols = st.columns(2)
            for idx, (key, prompt_data) in enumerate(filtered_prompts.items()):
                with cols[idx % 2]:
                    with st.container():
                        # Card header
                        is_selected = key in st.session_state.selected_prompts
                        
                        if is_selected:
                            st.markdown(f"**✓ {prompt_data['name']}**")
                        else:
                            st.markdown(f"**{prompt_data['name']}**")
                        
                        # Type badge
                        if key in DEFAULT_TEMPLATES:
                            st.caption("Default Template")
                        else:
                            st.caption("Custom Template")
                        
                        # Description
                        if prompt_data.get("description"):
                            st.caption(prompt_data["description"])
                        
                        # Tags
                        if prompt_data.get("tags"):
                            tags_str = " ".join([f"`{tag}`" for tag in prompt_data["tags"]])
                            st.markdown(tags_str)
                        
                        # Show template in expander
                        with st.expander("View Template"):
                            st.code(prompt_data["template"], language="text")
                            
                            # Token count
                            tokens = self.estimate_tokens(prompt_data["template"])
                            st.caption(f"Estimated tokens: ~{tokens}")
                        
                        # Action buttons
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("Edit", key=f"edit_{key}", use_container_width=True):
                                st.session_state.editing_prompt = key
                                st.rerun()
                        
                        with col_b:
                            if key not in DEFAULT_TEMPLATES:
                                if st.button("Delete", key=f"delete_{key}", use_container_width=True):
                                    del st.session_state.custom_prompts[key]
                                    st.rerun()
                        
                        st.markdown("---")
        else:
            st.info("No prompts match your filters")
    
    def get_selected_prompts(self) -> Dict[str, Dict]:
        """
        Get the selected prompts for use in surveys
        
        Returns:
            Dictionary of selected prompts
        """
        all_prompts = {**DEFAULT_TEMPLATES, **st.session_state.custom_prompts}
        selected = {}
        
        for key in st.session_state.selected_prompts:
            if key in all_prompts:
                selected[key] = all_prompts[key]
        
        return selected


def render_prompt_builder():
    """
    Main function to render the prompt builder component
    
    Returns:
        Dictionary of selected prompts
    """
    builder = PromptBuilder()
    return builder.render()


# Example usage
if __name__ == "__main__":
    st.set_page_config(
        page_title="Prompt Builder",
        page_icon="✏️",
        layout="wide"
    )
    
    selected_prompts = render_prompt_builder()
    
    if selected_prompts:
        st.sidebar.success(f"Selected {len(selected_prompts)} prompts")
        st.sidebar.json(selected_prompts)