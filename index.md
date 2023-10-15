# Welcome to the QA Project!

## About the Project
The Question Answering (QA) project aims to develop a robust model capable of understanding and answering user queries effectively. Utilizing various data sources and natural language processing techniques, the project seeks to create a system that can comprehend the context of questions and provide accurate, relevant answers.

## Project Overview
### Data Collection
- **Web Scraping**: Extracted relevant textual data from various web pages related to "borsa" using Python libraries like `requests` and `BeautifulSoup`.
- **API Usage**: Leveraged the Bing Search API to obtain URLs that serve as data sources for textual content.

### Data Preprocessing
- **Text Cleaning**: Employed regular expressions and string manipulation techniques to clean and standardize the textual data.
- **Tokenization and Stop-word Removal**: Utilized `nltk` for tokenization and custom stop-word removal to refine the data.

### Model Development
- **Question Identification**: Implemented regex patterns and NLP techniques to identify and extract questions from the textual data.
- **Spacy**: Utilized the `tr_core_news_trf` model from Spacy for various NLP tasks, such as named entity recognition and dependency parsing.

## Challenges Encountered
- **Data Limitations**: Encountered challenges in obtaining a substantial amount of question-answer pairs from the scraped data.
- **Technical Issues**: Faced difficulties with certain NLP tools, such as Zemberek-NLP, due to Java-related issues in the environment.

## Future Steps
- **Incorporating TS Corpus V2**: Plan to utilize the TS Corpus V2 dataset, which encompasses various grammatical categories in Turkish, to enhance the model's training and validation phases.
- **Deep Learning Integration**: Explore the integration of deep learning models like BERT for improved question understanding and answer generation.
- **Model Optimization**: Continuous refinement of the model to enhance its accuracy and reliability in providing answers.

## Get Involved!
We welcome contributions and collaborations! Feel free to fork the repository, submit issues, or send pull requests.

## Contact
For any inquiries or further discussion regarding the project, please [contact us](mailto:your_email@example.com).

