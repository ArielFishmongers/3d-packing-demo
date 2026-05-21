"""
Streamlit Cloud entry point. Delegates to the demo dashboard.

The ``if __name__ == "__main__"`` guard is load-bearing: multiprocessing
spawn workers re-import this script as ``__mp_main__`` to rebuild the
parent's main module. Without the guard they would re-run the dashboard
in bare mode, flooding logs with ScriptRunContext warnings.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).parent / "examples" / "dashboard.py"),
        run_name="__main__",
    )
