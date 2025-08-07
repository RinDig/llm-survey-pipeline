"""
Secure API Key Management Component for LLM Survey Pipeline
Provides a Streamlit interface for secure API key input and validation
"""

import streamlit as st
import asyncio
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime

# Import API clients for testing connections
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from llamaapi import LlamaAPI

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manages API keys securely in session state with validation"""
    
    # Provider configuration
    PROVIDERS = {
        "OpenAI": {
            "env_key": "OPENAI_API_KEY",
            "display_name": "OpenAI (GPT-4)",
            "help_url": "https://platform.openai.com/api-keys",
            "description": "Powers GPT-4 and other OpenAI models",
            "test_model": "gpt-4o",
            "icon": "🤖"
        },
        "Anthropic": {
            "env_key": "ANTHROPIC_API_KEY", 
            "display_name": "Anthropic (Claude)",
            "help_url": "https://console.anthropic.com/account/keys",
            "description": "Powers Claude 3.5 Sonnet and other Anthropic models",
            "test_model": "claude-3-5-sonnet-20241022",
            "icon": "🧠"
        },
        "Llama": {
            "env_key": "LLAMA_API_KEY",
            "display_name": "Llama API",
            "help_url": "https://www.llama-api.com",
            "description": "Access to Llama 3.1 models",
            "test_model": "llama3.1-70b",
            "icon": "🦙"
        },
        "Grok": {
            "env_key": "XAI_API_KEY",
            "display_name": "X.AI (Grok)",
            "help_url": "https://x.ai/api",
            "description": "Powers Grok-2 and other X.AI models",
            "test_model": "grok-2-latest",
            "icon": "✨",
            "base_url": "https://api.x.ai/v1"
        },
        "DeepSeek": {
            "env_key": "DEEPSEEK_API_KEY",
            "display_name": "DeepSeek",
            "help_url": "https://platform.deepseek.com/api_keys",
            "description": "Access to DeepSeek V3 models",
            "test_model": "deepseek-v3",
            "icon": "🔍"
        }
    }
    
    def __init__(self):
        """Initialize the API key manager with session state"""
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables for API keys"""
        if "api_keys" not in st.session_state:
            st.session_state.api_keys = {}
        
        if "api_key_validation" not in st.session_state:
            st.session_state.api_key_validation = {}
        
        if "last_validation_time" not in st.session_state:
            st.session_state.last_validation_time = {}
    
    async def _test_openai_connection(self, api_key: str, base_url: Optional[str] = None) -> Tuple[bool, str]:
        """Test OpenAI or compatible API connection"""
        try:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            response = await client.chat.completions.create(
                model="gpt-4o" if not base_url else "grok-2-latest",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                temperature=0
            )
            return True, "Connection successful"
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                return False, "Invalid API key"
            elif "429" in error_msg:
                return False, "Rate limit exceeded (key is valid)"
            elif "model" in error_msg.lower():
                return False, "Model access issue (key may be valid)"
            else:
                return False, f"Connection failed: {error_msg[:100]}"
    
    async def _test_anthropic_connection(self, api_key: str) -> Tuple[bool, str]:
        """Test Anthropic API connection"""
        try:
            client = AsyncAnthropic(api_key=api_key)
            response = await client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                temperature=0
            )
            return True, "Connection successful"
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                return False, "Invalid API key"
            elif "429" in error_msg:
                return False, "Rate limit exceeded (key is valid)"
            else:
                return False, f"Connection failed: {error_msg[:100]}"
    
    def _test_llama_connection(self, api_key: str, model: str) -> Tuple[bool, str]:
        """Test Llama API connection (synchronous)"""
        try:
            client = LlamaAPI(api_key)
            response = client.run({
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
                "temperature": 0
            })
            if response and response.json():
                return True, "Connection successful"
            else:
                return False, "Connection failed: No response"
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                return False, "Invalid API key"
            else:
                return False, f"Connection failed: {error_msg[:100]}"
    
    async def test_connection(self, provider: str, api_key: str) -> Tuple[bool, str]:
        """Test API connection for a specific provider"""
        if not api_key:
            return False, "No API key provided"
        
        provider_config = self.PROVIDERS.get(provider)
        if not provider_config:
            return False, f"Unknown provider: {provider}"
        
        try:
            if provider == "OpenAI":
                return await self._test_openai_connection(api_key)
            elif provider == "Anthropic":
                return await self._test_anthropic_connection(api_key)
            elif provider == "Grok":
                base_url = provider_config.get("base_url")
                return await self._test_openai_connection(api_key, base_url)
            elif provider in ["Llama", "DeepSeek"]:
                # Llama API is synchronous, so we run it in executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, 
                    self._test_llama_connection, 
                    api_key, 
                    provider_config["test_model"]
                )
            else:
                return False, f"Test not implemented for {provider}"
        except Exception as e:
            return False, f"Test failed: {str(e)[:100]}"
    
    def render_security_notice(self):
        """Render security notice and warnings"""
        with st.expander("🔒 **Security & Privacy Information**", expanded=False):
            st.markdown("""
            ### Your API Keys Are Safe
            
            - **Session-Only Storage**: API keys are stored only in your browser session
            - **Never Persisted**: Keys are never saved to disk or database
            - **Server-Side Processing**: All API calls are made server-side
            - **Encrypted Connection**: Use HTTPS in production for secure transmission
            - **Session Cleanup**: Keys are cleared when you close the browser
            
            ### Best Practices
            
            1. **Never commit API keys** to version control
            2. **Use environment variables** for production deployments
            3. **Rotate keys regularly** through your provider's dashboard
            4. **Set usage limits** with your API provider
            5. **Monitor usage** to detect any unauthorized access
            
            ### Need Help Getting API Keys?
            
            Click the "Get API Key" links next to each provider for instructions.
            """)
    
    def render_provider_input(self, provider: str, col):
        """Render input field for a single provider"""
        config = self.PROVIDERS[provider]
        
        with col:
            # Provider header with icon
            st.markdown(f"### {config['icon']} {config['display_name']}")
            st.caption(config['description'])
            
            # Current status
            current_key = st.session_state.api_keys.get(config['env_key'], "")
            is_validated = st.session_state.api_key_validation.get(provider, False)
            
            # Status indicator
            if current_key:
                if is_validated:
                    st.success("✅ Key validated", icon="✅")
                else:
                    st.warning("⚠️ Key provided but not validated", icon="⚠️")
            else:
                st.info("ℹ️ No key provided", icon="ℹ️")
            
            # API key input
            key_input = st.text_input(
                f"API Key",
                value=current_key,
                type="password",
                key=f"input_{provider}",
                help=f"Your {config['display_name']} API key",
                placeholder="sk-..."
            )
            
            # Buttons row
            col1, col2 = st.columns(2)
            
            with col1:
                st.link_button(
                    "Get API Key",
                    config['help_url'],
                    use_container_width=True,
                    type="secondary"
                )
            
            with col2:
                if st.button(
                    "Test Connection",
                    key=f"test_{provider}",
                    disabled=not key_input,
                    use_container_width=True,
                    type="primary" if key_input else "secondary"
                ):
                    return provider, key_input
            
            # Update session state if key changed
            if key_input != current_key:
                st.session_state.api_keys[config['env_key']] = key_input
                st.session_state.api_key_validation[provider] = False
        
        return None, None
    
    def render_bulk_actions(self):
        """Render bulk action buttons"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Test All Keys", use_container_width=True, type="primary"):
                return "test_all"
        
        with col2:
            if st.button("🗑️ Clear All Keys", use_container_width=True, type="secondary"):
                return "clear_all"
        
        with col3:
            # Count of valid keys
            valid_count = sum(st.session_state.api_key_validation.values())
            total_count = len(self.PROVIDERS)
            st.metric("Valid Keys", f"{valid_count}/{total_count}")
        
        return None
    
    def clear_all_keys(self):
        """Clear all API keys from session state"""
        st.session_state.api_keys = {}
        st.session_state.api_key_validation = {}
        st.session_state.last_validation_time = {}
    
    def get_validated_keys(self) -> Dict[str, str]:
        """Get dictionary of validated API keys"""
        validated = {}
        for provider, config in self.PROVIDERS.items():
            if st.session_state.api_key_validation.get(provider, False):
                key = st.session_state.api_keys.get(config['env_key'])
                if key:
                    validated[config['env_key']] = key
        return validated
    
    def get_all_keys(self) -> Dict[str, str]:
        """Get all provided API keys (validated or not)"""
        return st.session_state.api_keys.copy()
    
    def render(self):
        """Main render method for the API key manager component"""
        st.title("🔐 API Key Management")
        st.markdown("Configure your LLM provider API keys securely. Keys are only stored in your browser session.")
        
        # Security notice
        self.render_security_notice()
        
        st.divider()
        
        # Provider inputs in a 2-column layout
        st.subheader("Provider Configuration")
        
        # Track test requests
        test_requests = []
        
        # First row: OpenAI and Anthropic
        col1, col2 = st.columns(2)
        test_req = self.render_provider_input("OpenAI", col1)
        if test_req[0]:
            test_requests.append(test_req)
        
        test_req = self.render_provider_input("Anthropic", col2)
        if test_req[0]:
            test_requests.append(test_req)
        
        # Second row: Llama and Grok
        col1, col2 = st.columns(2)
        test_req = self.render_provider_input("Llama", col1)
        if test_req[0]:
            test_requests.append(test_req)
        
        test_req = self.render_provider_input("Grok", col2)
        if test_req[0]:
            test_requests.append(test_req)
        
        # Third row: DeepSeek (single column)
        col1, _ = st.columns(2)
        test_req = self.render_provider_input("DeepSeek", col1)
        if test_req[0]:
            test_requests.append(test_req)
        
        st.divider()
        
        # Bulk actions
        st.subheader("Bulk Actions")
        action = self.render_bulk_actions()
        
        # Handle test requests
        if test_requests:
            for provider, api_key in test_requests:
                with st.spinner(f"Testing {provider} connection..."):
                    try:
                        # Run async test
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        success, message = loop.run_until_complete(
                            self.test_connection(provider, api_key)
                        )
                        loop.close()
                        
                        # Update validation status
                        st.session_state.api_key_validation[provider] = success
                        st.session_state.last_validation_time[provider] = datetime.now()
                        
                        # Show result
                        if success:
                            st.success(f"✅ {provider}: {message}")
                        else:
                            st.error(f"❌ {provider}: {message}")
                    except Exception as e:
                        st.error(f"❌ {provider}: Error during test - {str(e)[:100]}")
                        st.session_state.api_key_validation[provider] = False
        
        # Handle bulk actions
        if action == "test_all":
            with st.spinner("Testing all provided keys..."):
                results = []
                for provider, config in self.PROVIDERS.items():
                    api_key = st.session_state.api_keys.get(config['env_key'])
                    if api_key:
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            success, message = loop.run_until_complete(
                                self.test_connection(provider, api_key)
                            )
                            loop.close()
                            
                            st.session_state.api_key_validation[provider] = success
                            st.session_state.last_validation_time[provider] = datetime.now()
                            results.append((provider, success, message))
                        except Exception as e:
                            results.append((provider, False, f"Error: {str(e)[:50]}"))
                            st.session_state.api_key_validation[provider] = False
                
                # Show results
                if results:
                    st.write("### Test Results")
                    for provider, success, message in results:
                        if success:
                            st.success(f"✅ {provider}: {message}")
                        else:
                            st.error(f"❌ {provider}: {message}")
                else:
                    st.warning("No API keys to test")
        
        elif action == "clear_all":
            # Confirmation dialog
            if st.session_state.get("confirm_clear", False):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Confirm Clear", type="primary", use_container_width=True):
                        self.clear_all_keys()
                        st.success("All API keys cleared")
                        st.session_state.confirm_clear = False
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", type="secondary", use_container_width=True):
                        st.session_state.confirm_clear = False
                        st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Are you sure you want to clear all API keys?")
                st.rerun()
        
        # Summary section
        st.divider()
        st.subheader("Configuration Summary")
        
        validated_keys = self.get_validated_keys()
        all_keys = self.get_all_keys()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Keys Provided", len(all_keys))
        with col2:
            st.metric("Validated Keys", len(validated_keys))
        with col3:
            # Models available
            available_models = []
            for provider in st.session_state.api_key_validation:
                if st.session_state.api_key_validation[provider]:
                    available_models.append(provider)
            st.metric("Available Models", len(available_models))
        
        if available_models:
            st.success(f"Ready to use: {', '.join(available_models)}")
        else:
            st.info("Add and validate API keys to enable model access")
        
        return validated_keys


def render_api_key_manager() -> Dict[str, str]:
    """
    Convenience function to render the API key manager and return validated keys
    
    Returns:
        Dictionary of validated API keys in format expected by backend
    """
    manager = APIKeyManager()
    return manager.render()


# Example usage in a Streamlit app
if __name__ == "__main__":
    st.set_page_config(
        page_title="API Key Manager",
        page_icon="🔐",
        layout="wide"
    )
    
    # Render the component
    validated_keys = render_api_key_manager()
    
    # Display the validated keys (for demo purposes)
    if validated_keys:
        st.divider()
        st.subheader("Validated API Keys (Demo Output)")
        st.json(validated_keys)