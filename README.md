# cv-job-matcher
IR-driven job matching based on your CV.

## Setting up the dependencies
`brew install pkg-config poppler python` + Python deps.

You need to provide a valid OpenAI key in a `key.yaml` file at the root of the project. It should have the following form:
```yaml
openai_key: <your key here>
```

## Starting the demo application
A web-based interface is bundled with the project. To use it, after setting up the dependencies, execute the following command from the project root directory:
```bash
streamlit run app.py
```
The application will pop up in your default browser and using it should be straightforward: you can upload any CV in PDF format and the application will return a list of jobs that matches the CV, along with scores and improvement recommendations for the jobs dynamically. This works after the preprocessed data has been generated, if you don't have the preprocessed data, refer to the "Generating the preprocessed job data" section.

TODO: dependencies, generating preprocessed job data...

## Generating the preprocessed job data

run `Job_description_embeddings.ipynb` to parse jobs and the body parsing. 
If job_openings_completed.json exists or `REPARSE_JOBS` in notebook is true the notebook will use gpt-3.5-turbo to parse the body of the jobs into appropriate fields. 
If an error occurs during parsing a specific job the fields will stay null and the job will additionally be written to the `job_openings_full_failed_{batch}.json`.
You can find the batch files in `job_openings ` and the consolidated jobs in `job_openings_completed.json`.