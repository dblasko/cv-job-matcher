import streamlit as st
from streamlit_lottie import st_lottie_spinner, st_lottie
import time
import json
import yaml
import requests


from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import job_description_embedding.JobMatchingBaseline as JobMatchingBaseline
import job_description_embedding.JobMatchingFineGrained as JobMatchingFineGrained

import cv_parsing.ResumeParser as ResumeParser
import job_description_embedding.JobMatchingBaseline as JobMatchingBaseline
from job_description_embedding.JobMatchingIdealJob import JobMatchingIdealJob
from job_description_embedding.CustomFakeLLM import CustomFakeLLM

LAYOUT_WIDE = False
FAKE_LLM = False
FAKE_REPONSE_COUNT = 10000
MAX_RESPONSE_TOKENS = 800

BASELINE_ENGINE = "Baseline - embedding full content"
IDEAL_ENGINE = "GPT Generated ideal job is matched to jobs"
FINE_GRAINED_ENGINE = (
    "Fine-grained - corresponding information of CV/job is matched & weighted in score"
)


def _get_fake_job(counter: str):
    return json.dumps(
        {
            "title": f"title-{counter}",
            "company": f"company-{counter}",
            "posted_date": f"posted_date-{counter}",
            "job_reference": f"job_reference-{counter}",
            "req_number": f"req_number-{counter}",
            "url": f"url-{counter}",
            "body": f"body-{counter}",
            "city": f"city-{counter}",
            "state": f"state-{counter}",
            "country": f"country-{counter}",
            "location": f"location-{counter}",
            "function": f"function-{counter}",
            "logo": f"logo-{counter}",
            "jobtype": f"jobtype-{counter}",
            "education": f"education-{counter}",
            "experience": f"experience-{counter}",
            "salary": f"salary-{counter}",
            "requiredlanguages": f"requiredlanguages-{counter}",
            "requiredskills": f"requiredskills-{counter}",
        }
    )


@st.cache_resource
def prepare_matching_engines():
    baseline = JobMatchingBaseline.JobMatchingBaseline(None)
    baseline.load_embeddings(
        "job_description_embedding/embeddings/saved_embeddings.pkl"
    )
    baseline.create_embedding_index()

    llm = (
        CustomFakeLLM(responses=[_get_fake_job(i) for i in range(FAKE_REPONSE_COUNT)])
        if FAKE_LLM
        else ChatOpenAI(
            openai_api_key=load_openai_key(),
            max_tokens=MAX_RESPONSE_TOKENS,
            model="gpt-3.5-turbo",
        )
    )

    ideal_engine = JobMatchingIdealJob(llm=llm)
    ideal_engine.load_embeddings(
        "job_description_embedding/embeddings/saved_embeddings.pkl"
    )
    ideal_engine.create_embedding_index()
    finegrained = JobMatchingFineGrained.JobMatchingFineGrained(None)
    finegrained.load_embeddings()
    # TODO: prepare other engines
    engines = {
        BASELINE_ENGINE: baseline,
        IDEAL_ENGINE: ideal_engine,
        FINE_GRAINED_ENGINE: finegrained,
    }
    return engines


def load_openai_key():
    with open("key.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config["openai_key"]


# Dummy functions for parsing and matching. You can replace them with your actual implementations.
def parse_pdf(file):
    parser = ResumeParser.ResumeParser(load_openai_key())
    parsed_cv = parser.pdf2string(file)
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
        selected_engine = st.selectbox(
            "Select a recommendation engine",
            [
                BASELINE_ENGINE,
                IDEAL_ENGINE,
                FINE_GRAINED_ENGINE,
            ],
        )
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
            # print(parsed_cv)
            # recommendations = matcher.match(parsed_cv)

            job_matching_engines = prepare_matching_engines()
            job_matching_engine = job_matching_engines[selected_engine]

            scores, job_offers = None, []
            with get_openai_callback() as call_logs:
                scores, job_offers = job_matching_engine.match_jobs(
                    str(parsed_cv), load_openai_key(), k=1000
                )
                print(call_logs, "\n")

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
