import os
import sys

from smart_analytics.core.data import Data
from smart_analytics.core.smarteda import SmartEDA
from smart_analytics.core.timeseries import SmartTimeSeries

def launch_ux():
    """Entrypoint function for the console script."""
    import streamlit.web.cli as stcli
    
    current_dir = os.path.dirname(__file__)
    # Points to src/smart_analytics/ui/app.py
    app_path = os.path.join(current_dir, "ui", "app.py")
    
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())

__all__ = ["Data", "SmartEDA", "TimeSeries", "launch_ux"]