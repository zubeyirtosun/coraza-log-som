import streamlit as st

def initialize_session_state():
    state_vars = {
        'df': None,
        'X': None,
        'som': None,
        'summary_df': None,
        'grid_size': None
    }
    for key, value in state_vars.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session_state():
    st.session_state.clear()
    initialize_session_state()
