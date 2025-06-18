import streamlit as st

def initialize_session_state():
    """
    Session state değişkenlerini başlatır
    """
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'X' not in st.session_state:
        st.session_state.X = None
    if 'som' not in st.session_state:
        st.session_state.som = None
    if 'som_weights_reshaped' not in st.session_state:
        st.session_state.som_weights_reshaped = None
    if 'meta_clusters' not in st.session_state:
        st.session_state.meta_clusters = None
    if 'direct_meta_clusters' not in st.session_state:
        st.session_state.direct_meta_clusters = None
    if 'direct_meta_labels' not in st.session_state:
        st.session_state.direct_meta_labels = None
    if 'direct_meta_metrics' not in st.session_state:
        st.session_state.direct_meta_metrics = None
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = None
    if 'meta_clustering_done' not in st.session_state:
        st.session_state.meta_clustering_done = False
    if 'som_done' not in st.session_state:
        st.session_state.som_done = False
    if 'comparison_type' not in st.session_state:
        st.session_state.comparison_type = "Doğrudan K-means vs SOM"
    if 'summary_df' not in st.session_state:
        st.session_state.summary_df = None
    if 'grid_size' not in st.session_state:
        st.session_state.grid_size = None
    if 'df_meta' not in st.session_state:
        st.session_state.df_meta = None
    if 'optimal_k_results' not in st.session_state:
        st.session_state.optimal_k_results = None
    if 'optimal_k' not in st.session_state:
        st.session_state.optimal_k = None
    if 'stability_results' not in st.session_state:
        st.session_state.stability_results = None
    if 'alternative_clustering_results' not in st.session_state:
        st.session_state.alternative_clustering_results = None
    if 'dimensionality_reduction_results' not in st.session_state:
        st.session_state.dimensionality_reduction_results = None
    if 'cross_validation_results' not in st.session_state:
        st.session_state.cross_validation_results = None
    if 'pdf_report' not in st.session_state:
        st.session_state.pdf_report = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'advanced_analysis_results' not in st.session_state:
        st.session_state.advanced_analysis_results = None


def reset_session_state():
    for key in list(st.session_state.keys()):
        if key not in ['_is_running', '_streamlit_session_random']:
            del st.session_state[key]
    initialize_session_state()
