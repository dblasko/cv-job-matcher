import streamlit as st
from streamlit_lottie import st_lottie_spinner, st_lottie
import time
import requests

LAYOUT_WIDE = False


# Dummy functions for parsing and matching. You can replace them with your actual implementations.
def parse_pdf(file):
    return "Parsed CV content"


def match_jobs(parsed_cv):
    job_offers = [
        {
            "title": "Software Engineer",
            "content": "We are looking for a software engineer...",
            "match_percentage": 90,
            "comments": ["Good programming skills", "Strong experience"],
        },
        {
            "title": "Software Engineer",
            "content": "We are looking for a software engineer...",
            "match_percentage": 80,
            "comments": ["Good programming skills", "Strong experience"],
        },
        # Add more job offers here
    ]
    return job_offers


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
            value=50,
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

            time.sleep(2)  # Simulate processing time
            parsed_cv = parse_pdf(uploaded_file)

            job_offers = match_jobs(parsed_cv)
            expander = cv_parsed_holder.expander(
                label=f"💡 **Transparency notice**: this is the information we have extracted from your CV.",
                expanded=False,
            )
            with expander:
                st.markdown(
                    f"""
                #### Skills
                - Skill 1 
                - Skill 2
                #### Experience  
                - Experience 1
                - Experience 2
                ### Education
                
                #### Languages
                
                #### Location
                
                #### Other          
                """
                )

            loading_text.empty()

        job_offers_filtered = [
            offer
            for offer in job_offers
            if offer["match_percentage"] >= match_threshold
        ]
        if not job_offers_filtered:
            st.warning(
                "No job offers matched, verify the file you uploaded or adjust the match percentage threshold!"
            )
        else:
            st.success("Job offers matched!")

            for index, offer in enumerate(job_offers_filtered):
                expander = st.expander(
                    label=f"{offer['title']} ({offer['match_percentage']}% match)",
                    expanded=False,
                )
                with expander:
                    st.write(offer["content"])
                    st.subheader("Match score: " + str(offer["match_percentage"]))
                    st.subheader("Comments:")
                    st.write("\n".join(offer["comments"]))


if __name__ == "__main__":
    main()
