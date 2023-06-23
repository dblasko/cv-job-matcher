import os
import json
from json.decoder import JSONDecodeError
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models.base import BaseChatModel
from langchain import PromptTemplate
import hashlib
import re
from job_description_embedding.JobMatchingBaseline import JobMatchingBaseline
from printer import eprint


# from dotenv import load_dotenv


class JobMatchingIdealJob(JobMatchingBaseline):
    def __init__(
            self,
            embeddings: HuggingFaceEmbeddings, 
            llm: BaseChatModel, cache_dir: str='.query_cache', 
            ideal_job_fields=[
                    'title'
                    'company',
                    'posted_date',
                    'body',
                    'city',
                    'state',
                    'country',
                    'location',
                    'function',
                    'jobtype',
                    'education',
                    'experience',
                    'salary',
                    'requiredlanguages',
                    'requiredskills'
                ],
            job_fields=[
            'title',
            'company',
            'posted_date',
            'job_reference',
            'req_number',
            'url',
            'body',
            'city',
            'state',
            'country',
            'location',
            'function',
            'logo',
            'jobtype',
            'education',
            'experience',
            'salary',
            'requiredlanguages',
            'requiredskills'
            ]
            ):
        super().__init__(embeddings=embeddings)
        self.llm = llm
        self.cache_dir = cache_dir
        self.sha256 = hashlib.sha256
        self.job_fields = job_fields

        self.prompt = PromptTemplate.from_template(
            '{cv}\nWhat would the Ideal job for this CV look like ? Extract the information in this CV into a valid flatt JSON object, parsable by json.loads().' +\
                ' The JSON represents a job and consists of the fields ' + ', '.join(ideal_job_fields) + \
                ' and fill in their content from this CV. Set the variable to null if the information is not derivable.' +\
                ' Reply with just the JSON object, keep the attribute values short and if appropriate in keywords.'
            )

    def match_jobs(self, query, k=5):
        query_d = self._get_ideal_job(query=query)
        if query_d is None:
            return (None, [])
        
        query_d = dict({k: None for k in self.job_fields if k not in query_d}, **query_d)
        query_result = self.embedder.embed_query(query_d)
        query_result = np.array(query_result)
        distances, neighbors = self.index.search(
            query_result.reshape(1, -1).astype(np.float32), k
        )

        scores = [distance for distance in distances[0]]
        # Normalize scores to be between 0 and 100
        scores = [100 * (1 - score / max(scores)) for score in scores]

        return (scores, [self.strings[neighbor] for neighbor in neighbors[0]])
    
    def _parse_json(self, response) -> dict | None:
        try:
            return json.loads(re.sub(r"(?<=\w)\n(?=\w)", "\\\\n", response.generations[0][0].text))
        except JSONDecodeError:
            eprint('Couldn\'t parse:', response.generations[0][0].text)
            return None
    
    def _get_ideal_job(self, query: str) -> dict | None:
        directory = os.path.join(os.getcwd(), self.cache_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f'ideal_job_cv-{self.sha256(query).hexdigest()}' + ".json")
        
        if not os.path.exists(file_path):
            try:
                prompt = self.prompt.format_prompt(cv=query)
                ideal_job = self._parse_json(self.llm.generate(messages=[prompt.to_messages()]))
                if ideal_job is not None:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(ideal_job, f)
            except Exception as err:
                eprint('got exception:', err)
                return None
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as j:
                return json.load(j)
        
        return None