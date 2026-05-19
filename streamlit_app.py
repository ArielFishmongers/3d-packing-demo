"""
Streamlit Cloud entry point. Delegates to the demo dashboard.
"""
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).parent / "examples" / "dashboard.py"),
    run_name="__main__",
)
