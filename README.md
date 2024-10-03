# 📝 AI-Powered Resume Parser and Classifier

Welcome to the AI-Powered Resume Parser and Classifier project! This application leverages advanced Natural Language Processing (NLP) techniques and Large Language Models (LLMs) to automate the extraction and classification of information from resumes. It's designed to streamline the recruitment process by accurately parsing resumes and categorizing candidates based on their skills and experiences.

## 📌 Table of Contents
- [Project Aim](#project-aim)
- [Project Structure](#project-structure)
- [Detailed Explanation](#detailed-explanation)
- [Technologies and Skills Used](#technologies-and-skills-used)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Running the Application](#running-the-application)
- [How It Can Be Used for Other Applications](#how-it-can-be-used-for-other-applications)
- [Contribution](#contribution)
- [License](#license)
- [Contact](#contact)


## 🎯 Project Aim

The primary goal of this project is to automate the resume screening process by:
- Extracting key information such as personal details, education, work experience, and skills from resumes.
- Classifying resumes into predefined job categories using Large Language Models.
- Providing a web interface for users to upload resumes and view parsed information along with predicted job roles.

## 📂 Project Structure

### app/ 📁
- `__init__.py`: Indicates that app is a Python package.
- `main.py`: The entry point of the application. Contains the Flask web server that handles routes and user interactions.
- `config.py`: Configuration settings for the application, such as model paths and hyperparameters.
- `data_preprocessing.py`: Functions for cleaning and preprocessing text data from resumes.
- `model_training.py`: Scripts to train the classification model using preprocessed data.
- `resume_parsing.py`: Logic to extract text from resumes and prepare it for parsing.
- `utils.py`: Utility functions and helpers used across the application.

### models/ 📁
- `__init__.py`: Indicates that models is a Python package.
- `bert_classifier.py`: Contains the `BertClassifier` class that wraps around the BERT model for classification tasks.
- `resume_parser_model.py`: Defines the `ResumeParser` class responsible for parsing resumes and extracting entities.

### data/ 📁
- `resumes/`: Directory containing sample resume PDFs for testing and development.
- `datasets/`: Contains datasets like `resumes.csv` used for training and evaluating the model.
- `skills.txt`: A text file with a list of skills used by the parser to identify skills in resumes.

### tests/ 📁
- `__init__.py`: Indicates that tests is a Python package.
- `test_data_preprocessing.py`: Unit tests for data preprocessing functions.
- `test_model_training.py`: Unit tests for the model training process.
- `test_resume_parsing.py`: Unit tests for the resume parsing logic.

### templates/ 📁
- `index.html`: The home page template where users can upload resumes.
- `result.html`: Displays the parsed resume information and classification results.

### Root Files
- `requirements.txt`: Lists all Python dependencies required for the project.
- `Dockerfile`: Instructions to containerize the application using Docker.
- `docker-compose.yml`: Defines services for Docker Compose to run the application.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: Documentation of the project (this file).
- `.github/workflows/ci_cd.yml`: Configuration for GitHub Actions to set up CI/CD pipelines.

## 💻 Technologies and Skills Used

### Programming and Scripting 🧑‍💻
- Python: Main programming language used for the application.
- Flask: Web framework for building the web application.
- PyTorch: Deep learning framework used for model development.
- Transformers: Hugging Face library for state-of-the-art NLP models.

### Natural Language Processing 🧠
- spaCy: Used for text processing and entity recognition.
- NLTK: Employed for text preprocessing tasks like tokenization and stop-word removal.
- BERT: Bidirectional Encoder Representations from Transformers for resume classification.
- Regular Expressions: For pattern matching in text (e.g., extracting emails and phone numbers).

### Data Handling and Storage 📊
- Pandas: For data manipulation and analysis.
- NumPy: Numerical computing library used for handling arrays.

### Containerization and Deployment 📦
- Docker: Containerization platform to package the application for deployment.
- Docker Compose: Tool for defining and running multi-container Docker applications.

### Continuous Integration/Continuous Deployment 🚀
- GitHub Actions: Automates testing and deployment workflows.
- Unit Testing: Using `unittest` framework to ensure code reliability.

### Version Control 🗂️
- Git: Version control system to track changes in the codebase.
- GitHub: Hosting service for Git repositories.

### Skills Demonstrated 🎓
- Data Engineering: Building data pipelines and preprocessing large datasets.
- Machine Learning: Training and deploying models for classification tasks.
- Software Development Best Practices: Modular co
