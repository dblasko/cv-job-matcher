import streamlit as st
from streamlit_lottie import st_lottie_spinner, st_lottie
import time
import json
import yaml
import requests
import job_description_embedding.JobMatching as JobMatching
import cv_parsing.ResumeParser as ResumeParser

LAYOUT_WIDE = False


@st.cache_resource
def prepare_matching_engine():
    job_matching_engine = JobMatching.JobMatching(None)
    job_matching_engine.load_embeddings(
        "job_description_embedding/embeddings/saved_embeddings.pkl"
    )
    job_matching_engine.create_embedding_index()
    return job_matching_engine


def load_openai_key():
    with open("key.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config["openai_key"]


# Dummy functions for parsing and matching. You can replace them with your actual implementations.
def parse_pdf(file):
    parser = ResumeParser.ResumeParser(load_openai_key())
    parsed_cv = parser.query_resume(file)
    return parsed_cv


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def main():
    st.set_page_config(
        page_title="CV-based Job Matching Tool",
        layout="centered" if not LAYOUT_WIDE else "wide",
    )
    st.title("👩‍🎓 CV-based Job Matching Tool")

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    col1, col2 = st.columns(2) if LAYOUT_WIDE else (st.container(), st.container())
    with col1:
        st.write(
            "Upload your CV and find the best job matches! <br/>**For ethical reasons, information about your age, gender or name will be ignored.**",
            unsafe_allow_html=True,
        )
        if LAYOUT_WIDE:
            cv_parsed_holder = st.empty()
    with col2:
        uploaded_file = st.file_uploader("Upload your CV (PDF format)", type=["pdf"])
        if not LAYOUT_WIDE:
            cv_parsed_holder = st.empty()
        match_threshold = st.slider(
            "Minimum match percentage to consider",
            min_value=0,
            max_value=100,
            value=25,
            step=5,
        )

    lottie_url = "https://assets3.lottiefiles.com/packages/lf20_hssvolmu.json"
    lottie_json = load_lottieurl(lottie_url)

    if uploaded_file is not None:
        with st_lottie_spinner(lottie_json, quality="high", height=250):
            col_1, col_2, col_3 = st.columns(3)
            with col_2:
                loading_text = st.empty()
                loading_text.write("*Matching your CV to offers ...*")

            parsed_cv = parse_pdf(uploaded_file)

            job_matching_engine = prepare_matching_engine()
            distances, job_offers = job_matching_engine.match_jobs(
                str(parsed_cv), k=1000
            )
            scores = [distance for distance in distances[0]]
            # Normalize scores to be between 0 and 100
            scores = [100 * (1 - score / max(scores)) for score in scores]

            expander = cv_parsed_holder.expander(
                label=f"💡 **Transparency notice**: this is the information we have extracted from your CV.",
                expanded=False,
            )
            with expander:
                st.write(parsed_cv)

            loading_text.empty()

        job_offers_filtered = [
            offer
            for idx, offer in enumerate(job_offers)
            if scores[idx] >= match_threshold
        ]
        if not job_offers_filtered:
            st.warning(
                "No job offers matched, verify the file you uploaded or adjust the match percentage threshold!"
            )
        else:
            st.success("Job offers matched!")

            for index, offer in enumerate(job_offers_filtered):
                expander = st.expander(
                    label=f"{offer['title']} ({round(scores[index],1)}% match)",
                    expanded=False,
                )
                with expander:
                    st.markdown(
                        f"""
                    ### {offer['title']}
                    #### General information
                    - **Company**: {offer['company']}
                    - **Function**: {offer['function']}
                    - **Job type**: {offer['jobtype']}
                    - **Location**: {offer['location']}, {offer['country']}
                    - **Posted date**: {offer['posted_date']}
                    - [**Posting URL**]({offer['url']})   
                    #### Original content
                    """
                    )
                    st.write(offer["body"], unsafe_allow_html=True)


if __name__ == "__main__":
    main()
