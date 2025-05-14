import streamlit as st

def initialize_session_state():
    state_vars = {
        'df': None,
        'X': None,
        'som': None,
        'summary_df': None,
        'grid_size': None,
        'som_weights_reshaped': None,
        'meta_clusters': None,
        'df_meta': None,
        'optimal_k_results': None,
        'optimal_k': None,
        'stability_results': None,
        'alternative_clustering_results': None,
        'dimensionality_reduction_results': None,
        'cross_validation_results': None,
        'pdf_report': None,
        'feature_names': None
    }
    for key, value in state_vars.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session_state():
    for key in list(st.session_state.keys()):
        if key not in ['_is_running', '_streamlit_session_random']:
            del st.session_state[key]
    initialize_session_state()
