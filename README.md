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


## Generating the preprocessed job data
