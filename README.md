# 📝 AI-Powered Resume Parser and Classifier

Welcome to the AI-Powered Resume Parser and Classifier project! This application leverages advanced Natural Language Processing (NLP) techniques and Large Language Models (LLMs) to automate the extraction and classification of information from resumes. It's designed to streamline the recruitment process by accurately parsing resumes and categorizing candidates based on their skills and experiences.
## 📌 Table of Contents

    🎯 Project Aim
    📂 Project Structure
    🔍 Detailed Explanation
    💻 Technologies and Skills Used
    🚀 Getting Started
        Prerequisites
        Installation
        Running the Application
    🌐 How It Can Be Used for Other Applications
    🤝 Contribution
    📄 License
    📞 Contact

## 🎯 Project Aim

The primary goal of this project is to automate the resume screening process by:

    Extracting key information such as personal details, education, work experience, and skills from resumes.
    Classifying resumes into predefined job categories using Large Language Models.
    Providing a web interface for users to upload resumes and view parsed information along with predicted job roles.


## 🔍 Detailed Explanation
### app/ 📁

    __init__.py: Indicates that app is a Python package.
    main.py: The entry point of the application. Contains the Flask web server that handles routes and user interactions.
    config.py: Configuration settings for the application, such as model paths and hyperparameters.
    data_preprocessing.py: Functions for cleaning and preprocessing text data from resumes.
    model_training.py: Scripts to train the classification model using preprocessed data.
    resume_parsing.py: Logic to extract text from resumes and prepare it for parsing.
    utils.py: Utility functions and helpers used across the application.

### models/ 📁

    __init__.py: Indicates that models is a Python package.
    bert_classifier.py: Contains the BertClassifier class that wraps around the BERT model for classification tasks.
    resume_parser_model.py: Defines the ResumeParser class responsible for parsing resumes and extracting entities.

### data/ 📁

    resumes/: Directory containing sample resume PDFs for testing and development.
    datasets/: Contains datasets like resumes.csv used for training and evaluating the model.
    skills.txt: A text file with a list of skills used by the parser to identify skills in resumes.

### tests/ 📁

    __init__.py: Indicates that tests is a Python package.
    test_data_preprocessing.py: Unit tests for data preprocessing functions.
    test_model_training.py: Unit tests for the model training process.
    test_resume_parsing.py: Unit tests for the resume parsing logic.

### templates/ 📁

    index.html: The home page template where users can upload resumes.
    result.html: Displays the parsed resume information and classification results.

### Root Files

    requirements.txt: Lists all Python dependencies required for the project.
    Dockerfile: Instructions to containerize the application using Docker.
    docker-compose.yml: Defines services for Docker Compose to run the application.
    .gitignore: Specifies files and directories to be ignored by Git.
    README.md: Documentation of the project (this file).
    .github/workflows/ci_cd.yml: Configuration for GitHub Actions to set up CI/CD pipelines.

## 💻 Technologies and Skills Used
### Programming and Scripting 🧑‍💻

    Python: Main programming language used for the application.
    Flask: Web framework for building the web application.
    PyTorch: Deep learning framework used for model development.
    Transformers: Hugging Face library for state-of-the-art NLP models.

### Natural Language Processing 🧠

    spaCy: Used for text processing and entity recognition.
    NLTK: Employed for text preprocessing tasks like tokenization and stop-word removal.
    BERT: Bidirectional Encoder Representations from Transformers for resume classification.
    Regular Expressions: For pattern matching in text (e.g., extracting emails and phone numbers).

### Data Handling and Storage 📊

    Pandas: For data manipulation and analysis.
    NumPy: Numerical computing library used for handling arrays.

### Containerization and Deployment 📦

    Docker: Containerization platform to package the application for deployment.
    Docker Compose: Tool for defining and running multi-container Docker applications.

### Continuous Integration/Continuous Deployment 🚀

    GitHub Actions: Automates testing and deployment workflows.
    Unit Testing: Using unittest framework to ensure code reliability.

### Version Control 🗂️

    Git: Version control system to track changes in the codebase.
    GitHub: Hosting service for Git repositories.

### Skills Demonstrated 🎓

    Data Engineering: Building data pipelines and preprocessing large datasets.
    Machine Learning: Training and deploying models for classification tasks.
    Software Development Best Practices: Modular code structure, documentation, and testing.
    DevOps: Containerization, CI/CD pipelines, and deployment strategies.
    Problem-Solving: Tackling complex challenges with innovative solutions.

## 🚀 Getting Started
Prerequisites

Ensure you have the following installed:

    Python 3.7+
    pip (Python package installer)
    Docker (if you plan to use Docker)
    Git (for version control)

Installation

    
git clone https://github.com/noushin94/resume_parser_classifier.git
cd resume_parser_classifier

Create a Virtual Environment



python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install Dependencies



pip install --upgrade pip
pip install -r requirements.txt

Download spaCy Model and NLTK Data



    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

## Running the Application


   Prepare the Dataset
        Place your dataset CSV file (resumes.csv) in data/datasets/.
        Ensure it has the columns Resume_str and Category.

    Train the Model

    
    python app/model_training.py

### Running the Web Application

    Start the Flask App

    

    python app/main.py

    Access the Application

    Open your browser and navigate to http://localhost:5000.

### Using Docker

    

   

docker build -t resume_parser_classifier .

Run the Docker Container



    docker run -p 5000:5000 resume_parser_classifier

## 🌐 How It Can Be Used for Other Applications

This project serves as a robust foundation for various applications:

    HR Automation Tools: Integrate the system into HR platforms to automate candidate screening.
    Job Matching Services: Enhance job portals by providing automatic resume parsing and job matching.
    Educational Platforms: Assist students in improving their resumes by providing automated feedback.
    Document Processing: Adapt the parsing logic to process other types of documents (e.g., cover letters, proposals).
    Custom Classification Tasks: Retrain the model on different datasets for tasks like sentiment analysis or topic classification.

## 🤝 Contribution

Contributions are welcome! Please follow these steps:

    Fork the Repository

    Create a Branch

    

git checkout -b feature/YourFeature

Commit Your Changes



git commit -m "Add your message"

Push to the Branch


    git push origin feature/YourFeature

    Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
📞 Contact

For any inquiries or feedback, please contact:

    Name: Noushin
    Email: Noushin_ahmadvand@yahoo.com
    

Thank you for checking out this project! We hope it provides valuable insights into the practical applications of NLP and machine learning in automating and enhancing recruitment processes. If you find this project helpful, please give it a ⭐ on GitHub!

😊 Happy Coding!
