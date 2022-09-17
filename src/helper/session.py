from streamlit.runtime.scriptrunner import get_script_run_ctx


def get_session_id():
    return get_script_run_ctx().session_id