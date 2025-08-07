"""Debug script to understand the deployment environment"""
import streamlit as st
import sys
import os
from pathlib import Path

st.title("Debug Information")

st.header("Current Working Directory")
st.code(os.getcwd())

st.header("Script Location")
st.code(__file__)

st.header("Parent Directory")
parent = Path(__file__).parent
st.code(str(parent))

st.header("Directory Contents")
for item in sorted(os.listdir(parent)):
    if os.path.isdir(parent / item):
        st.write(f"📁 {item}/")
    else:
        st.write(f"📄 {item}")

st.header("Python Path")
for i, path in enumerate(sys.path):
    st.code(f"{i}: {path}")

st.header("Can we import?")
try:
    # Try different import methods
    st.write("Trying: from frontend.components import api_key_manager")
    from frontend.components import api_key_manager
    st.success("✅ Success!")
except ImportError as e:
    st.error(f"❌ Failed: {e}")

try:
    # Try adding to path first
    st.write("Adding parent to sys.path and trying again...")
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from frontend.components import api_key_manager
    st.success("✅ Success with sys.path!")
except ImportError as e:
    st.error(f"❌ Still failed: {e}")

st.header("Frontend Directory")
frontend_path = parent / "frontend"
if frontend_path.exists():
    st.success(f"✅ Frontend exists at: {frontend_path}")
    st.write("Contents:")
    for item in sorted(os.listdir(frontend_path)):
        st.write(f"  - {item}")
else:
    st.error("❌ Frontend directory not found!")

st.header("Backend Directory")
backend_path = parent / "backend"
if backend_path.exists():
    st.success(f"✅ Backend exists at: {backend_path}")
    st.write("Contents:")
    for item in sorted(os.listdir(backend_path)):
        st.write(f"  - {item}")
else:
    st.error("❌ Backend directory not found!")