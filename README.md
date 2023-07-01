# cv-job-matcher
IR-driven job matching based on your CV.

## Setting up the environment
**System-level dependencies the packages require:**
- `pkg-config`
- `poppler`

Examples of how to install these dependencies on various platforms:
- Ubuntu: `sudo apt-get install pkg-config poppler-utils`
- Mac OS: `brew install pkg-config poppler`

**Python dependencies:**
The dependencies have been specified in a `requirements.txt` file. To install them, run the following command from the project root directory, ideally in a new envgironment:
```bash
pip install -r requirements.txt
```

You need to **provide a valid OpenAI key in a `key.yaml` file at the root of the project**. It should have the following form:
```yaml
openai_key: '<your key here>'
```

## Starting the demo application
A web-based interface is bundled with the project. Everything is run from that web interface. To use it, after setting up the dependencies, execute the following command from the project root directory:
```bash
streamlit run app.py
```
The application will pop up in your default browser and using it should be straightforward: you can upload any CV in PDF format and the application will return a list of jobs that matches the CV, along with scores and improvement recommendations for the jobs dynamically. **This works after the preprocessed data has been generated, if you do not have the preprocessed data, refer to the "Generating the preprocessed job data" section**.

## Generating the preprocessed job data

To limit the inference cost when a user performs a CV-matching requests, the web-application only uses pre-computed embeddings for the documents (job postings) and only embeds the query (CV or ideal job based on the CV) on the fly. Therefore, the first time you run the application, you will need to generate the preprocessed data locally.

The code comes with the pre-processed embedding (pre-processing code is also included): `job_description_embedding/job_openings.json` and `job_description_embedding/job_openings_completed.json`. The second file is the version of the first file where null fields of the job postings have been completed with GPT-3.5-turbo - we include the file to cut the costs as generating those is expensive. 

The locally create the embedding based on those files, run: 
```bash
python job_description_embedding/JobMatchingBaseline.py
python job_description_embedding/JobMatchinFineGrained.py
```

This results in a `job_description_embedding/embeddings` folder containing `saved_embeddings.pkl` and `fine_graind/...`. 

To re-generate the job posting null completions with GPT-3.5-turbo (expensive): run `Job_description_embeddings.ipynb` to parse jobs and the body parsing. 
If job_openings_completed.json exists or `REPARSE_JOBS` in notebook is true the notebook will use gpt-3.5-turbo to parse the body of the jobs into appropriate fields. 
If an error occurs during parsing a specific job the fields will stay null and the job will additionally be written to the `job_openings_full_failed_{batch}.json`.
You can find the batch files in `job_openings ` and the consolidated jobs in `job_openings_completed.json`.
