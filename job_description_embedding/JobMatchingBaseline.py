import os
import json
import xml.etree.ElementTree as ET
import faiss
import pickle
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

# from dotenv import load_dotenv


class JobMatchingBaseline:
    def __init__(self, embeddings: HuggingFaceEmbeddings):
        self.embedder = HuggingFaceEmbeddings()
        # load_dotenv()  # Load environment variables from .env file
        self.embeddings = None
        self.index = None
        self.strings = None

    def parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        jobs_list = []
        for job in root.findall("job"):
            job_dict = {
                "title": job.find("title").text,
                "company": job.find("company").text,
                "posted_date": job.find("posted_date").text,
                "job_reference": job.find("job_reference").text,
                "req_number": job.find("req_number").text,
                "url": job.find("url").text,
                "body": job.find("body").text,
                "city": job.find("city").text,
                "state": job.find("state").text,
                "country": job.find("country").text,
                "location": job.find("location").text,
                "function": job.find("function").text,
                "logo": job.find("logo").text,
                "jobtype": job.find("jobtype").text,
                "education": job.find("education").text,
                "experience": job.find("experience").text,
                "salary": job.find("salary").text,
                "requiredlanguages": job.find("requiredlanguages").text,
                "requiredskills": job.find("requiredskills").text,
            }
            jobs_list.append(job_dict)

        return jobs_list

    def xml_to_json(self, xml_file, json_output_file):
        jobs_list = self.parse_xml(xml_file)
        json_output = json.dumps(jobs_list, indent=4)

        with open(json_output_file, "w") as json_file:
            json_file.write(json_output)

    def create_embeddings(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

        strings = []
        for obj in data:
            string = json.dumps(obj)
            strings.append(string)

        doc_result = self.embedder.embed_documents(strings)

        index = faiss.index_factory(len(doc_result[0]), "Flat")
        index.train(doc_result)
        index.add(doc_result)
        self.index = index

        return index, strings

    def create_embedding_index(self):
        index = faiss.index_factory(len(self.embeddings[0]), "Flat")
        index.train(self.embeddings)
        index.add(self.embeddings)
        self.index = index

    def match_jobs(self, query, openai_key, k=5):
        query = """
DANIEL BLASKO daniel.blasko@insa-lyon.fr /in/daniel-blasko EDUCATION MSc. - Data Science | University of Technology Vienna, Austria MEng. - Computer Science | National Institute of Applied Sciences Lyon, France Higher National Diploma - Computer Science | IUT Lyon 1, France Baccalaureate in Sciences – with distinction | Jean Sturm, Strasbourg, France | 2022-2024 | 2020-2023 | 2018-2020 | 2016-2018 PROFESSIONAL EXPERIENCES INTERDISCIPLINARY PROJECT | IFT TU Wien – Vienna, Austria | MARCH-JULY 2023 DATA SCIENCE FOR IoT SECURITY – INTERN | Cisco - Lyon, France | MAY-SEPT. 2023 CONSULTANT | ETIC INSA Lyon (Junior business) - Lyon, France | 2020–2022 MOBILE DEVELOPMENT INTERN | Worldline Global - Lyon, France | JUNE-AUGUST 2021 Research project at the institute for manufacturing engineering and photonic technologies. Experimental evaluation of industrial machine data extraction methods using different OPC-UA endpoint implementations and industrial edge devices on Siemens and FANUC machines. On-premise network data-analysis and modelling for identification and tracking of components on industrial IoT networks in the Cisco Cyber Vision product. Created client-dataset analysis and reporting tools and integrated machine-learning algorithms in the product for network component classification and clustering. Understanding and formalizing client needs, conceiving and implementing mock-ups, exchanging with clients to support them in their IT projects. Cross-platform development in Flutter for two mobile banking applications. Analyzed the current in a presentation and written article synthetizing the tradeoffs of a transition to Flutter for new projects. SOFTWARE DEVELOPMENT INTERN | NatBraille & LIRIS lab - remote| APRIL-JULY 2020 Modelled, specified, and implemented a pedagogical web application to teach Braille. Organized and led user interviews, specified the requirements and modelized the architecture. Implemented Braille application in PHP and JavaScript, with key emphasis on web accessibility. R&D INTERNSHIP | Transchain - Strasbourg, France | JULY-AUGUST 2019 Developed an API, multiple GUIs and different system scripts in Golang for Transchain’s blockchain in a scrumbased environment. Heavy usage of Docker containerization and creation of multiple continuous integration scripts. Ensured a high test-coverage for every project. The created tools are used by clients & internally. SKILLS Languages: Data Science tools: Frameworks: Practical Python experience with ScikitR (tidyverse) learn, Keras, PyTorch, Prolog (fuzzy logic) OpenCV & data Java: Swing, Spring visualization libraries. Android Data intensive computing: Dart (Flutter) AirFlow, Hadoop, Spark. JavaScript, PHP Knowledge graphs: C, C++ (systems & parallel programming) experience with Protégé, OpenRefine, GraphDB, KG embeddings. Systems: Databases: Linux Windows, MacOS Docker containerization Computer Networks: Deployment: Gitlab CI, Jenkins CI, Github actions SQL, PLSQL, MongoDB General networking theory, experience with industrial networking protocols. Languages: French – Native, C2 Slovak – Native, C2 English – TOEIC 990, C1 German – DSD2, C1 certificate
            """
        query_result = self.embedder.embed_query(query)
        query_result = np.array(query_result)
        distances, neighbors = self.index.search(
            query_result.reshape(1, -1).astype(np.float32), k
        )

        scores = [distance for distance in distances[0]]
        # Normalize scores to be between 0 and 100
        scores = [100 * (1 - score / max(scores)) for score in scores]

        return (scores, [self.strings[neighbor] for neighbor in neighbors[0]])

    def save_embeddings(
        self,
        embeddings,
        saving_embeddings_file_name: str = os.getenv("SAVING_EMBEDDINGS_FILE_NAME"),
        saving_embeddings_directory: str = os.getenv("SAVING_EMBEDDINGS_DIRECTORY"),
    ) -> None:
        directory = os.path.join(os.getcwd(), saving_embeddings_directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, saving_embeddings_file_name + ".pkl")

        # Save embeddings to binary file
        with open(file_path, "wb") as f:
            pickle.dump(embeddings, f)

    def load_embeddings(self, embeddings_path) -> HuggingFaceEmbeddings:
        print("CALLED")
        with open(embeddings_path, "rb") as f:
            embeddings: HuggingFaceEmbeddings = pickle.load(f)

        print(type(embeddings))
        self.embeddings = embeddings

        with open("job_description_embedding/job_openings.json", "r") as f:
            strings = json.load(f)
        self.strings = strings
