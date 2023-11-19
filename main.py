import requests
import json
import time
import re
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import logging
import spacy

# Your Bing Search v7 API key
api_key = '901f376c293249aca48b26d516204427'

# Search query
query = 'borsa'

# API request headers
headers = {
    'Ocp-Apim-Subscription-Key': api_key,
}

# API request parameters
params = {
    'q': query,
    'count': 50,
    'offset': 0,
    'mkt': 'tr-TR',
    'safesearch': 'Moderate',
}

# API endpoint URL
url = 'https://api.bing.microsoft.com/v7.0/search'

# Make API request
response = requests.get(url, headers=headers, params=params)

# Check for request failure
if response.status_code != 200:
    print(f"Request failed with status code {response.status_code}: {response.text}")
else:
    # Parse response
    results = response.json()

    # Create a list to store URLs
    urls = []

    # Extract URLs from results and add them to the list
    for result in results.get('webPages', {}).get('value', []):
        urls.append(result['url'])
        print(f"URL: {result['url']}")
        print(f"Snippet: {result['snippet']}\n")

    # Now `urls` is a list containing all the URLs from the search results
    
# Define the base URL and user agent for the HTTP request
base_url = "https://eksisozluk1923.com/borsa--58544"
user_agent = 'Mozilla/5.0 (Platform; Security; OS-or-CPU; Localization; rv:1.4) Gecko/20030624 Netscape/7.1 (ax)'

# Define how many pages you want to scrape
start_page = 1
end_page = 56 # Change this to the number of pages you want to scrape
delay = 1  # Delay between page requests

# Initialize an empty list to store all text data
all_text_data = []

# Loop through the specified range of pages
for page_num in range(start_page, end_page + 1):
    # Construct the URL for the current page
    page_url = f"{base_url}?p={page_num}"
    print(f"Scraping page: {page_url}")

    # Send the HTTP request to the page
    response = requests.get(page_url, headers={'User-Agent': user_agent})
    
    # Check the status code before processing to avoid errors
    if response.status_code == 200:
        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find all div elements with the class 'content'
        entries = soup.find_all("div", {"class": "content"})
        for entry in entries:
            all_text_data.append(entry.text.strip())  # Append the text content of each entry to the list
        
        # Wait for a short while before requesting the next page to be polite to the server
        time.sleep(delay)
    else:
        print(f"Failed to retrieve page {page_num}: HTTP {response.status_code}")
        break     
        

# Define the base URL and user agent for the HTTP request
base_url = "https://eksisozluk1923.com/borsa-istanbul--3559807"
user_agent = 'Mozilla/5.0 (Platform; Security; OS-or-CPU; Localization; rv:1.4) Gecko/20030624 Netscape/7.1 (ax)'

# Define how many pages you want to scrape
start_page = 1
end_page = 2630 # Change this to the number of pages you want to scrape
delay = 2  # Delay between page requests

# Initialize an empty list to store all text data
all_text_data_ist = []

# Loop through the specified range of pages
for page_num in range(start_page, end_page + 1):
    # Construct the URL for the current page
    page_url = f"{base_url}?p={page_num}"
    print(f"Scraping page: {page_url}")

    # Send the HTTP request to the page
    response = requests.get(page_url, headers={'User-Agent': user_agent})
    
    # Check the status code before processing to avoid errors
    if response.status_code == 200:
        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find all div elements with the class 'content'
        entries = soup.find_all("div", {"class": "content"})
        for entry in entries:
            all_text_data_ist.append(entry.text.strip())  # Append the text content of each entry to the list
        
        # Wait for a short while before requesting the next page to be polite to the server
        time.sleep(delay)
    else:
        print(f"Failed to retrieve page {page_num}: HTTP {response.status_code}")
        break 
        
# Download NLTK data
nltk.download('punkt')

# Example text in Turkish
text = " ".join(all_text_data)  # Join the list into a single string

### 1. Cleaning
# Remove non-alphanumeric characters and convert to lowercase
# Replace newline characters with space
cleaned_text = text.replace('\n', ' ')

# Remove non-alphanumeric characters and convert to lowercase
cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)  # Keep only alphanumeric and spaces
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip().lower()  # Remove extra spaces and convert to lowercase

### 2. Tokenization
tokens = word_tokenize(cleaned_text)

### 3. Stop-word Removal
# Define the path to your stopwords file
stopwords_file_path = '../../trstop/test.txt'

# Read the stopwords file and create a list of stopwords
with open(stopwords_file_path, 'r', encoding='utf-8') as file:
    turkish_stopwords = [line.strip() for line in file]

# Remove stopwords
filtered_tokens = [token for token in tokens if token not in turkish_stopwords]

# Final preprocessed text
preprocessed_text = ' '.join(filtered_tokens)
print(preprocessed_text)        

# Concatenate all text data into a single string
merged_text_data = "\n".join(all_text_data)

# Find questions using regex
questions = re.findall(r'(?i)\b(nasıl|ne|neden|ne zaman|kim|hangi|kaç)[\w\s]*\?', merged_text_data)

# Print found questions
for question in questions:
    print(question)

# Load a language model
nlp = spacy.load('tr_core_news_trf')  # For Turkish, you might use 'tr_core_news_sm'

# Process the text
doc = nlp(preprocessed_text)

# Extract named entities
for ent in doc.ents:
    print(ent.text, ent.label_)   
    
def generate_qa_pairs(text):
    doc = nlp(text)
    qa_pairs = []

    for ent in doc.ents:
        # Extract the sentence as context
        context = next(sent.text for sent in doc.sents if ent.start_char >= sent.start_char and ent.end_char <= sent.end_char)
        
        # Match the entity type to the appropriate question word
        if ent.label_ == 'METHOD':  # This is a custom label you would have added
            question_word = 'nasıl'
        elif ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']:
            question_word = 'hangi'
        elif ent.label_ in ['DATE', 'TIME']:
            question_word = 'ne zaman'
        elif ent.label_ in ['NORP', 'LANGUAGE']:
            question_word = 'hangi'
        else:
            question_word = 'ne'
        
        # Replace the entity text with the question word to form the question
        question = context.replace(ent.text, question_word)
        
        # Create a dictionary for each Q&A pair
        qa_pair = {
            'context': context,
            'question': question + '?',
            'answer': ent.text
        }
        qa_pairs.append(qa_pair)
    
    return qa_pairs

# Suppose 'text' contains multiple paragraphs of your entire document
text = """
Your long text goes here.
"""

# Generate Q&A pairs
qa_pairs = generate_qa_pairs(preprocessed_text)

# Save as JSON
with open('qa_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

# Convert to DataFrame for CSV output
qa_df = pd.DataFrame(qa_pairs)

# Save as CSV
qa_df.to_csv('qa_dataset.csv', index=False)    

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

# Suppose we have a DataFrame with contexts, questions, and answers
contexts = qa_df.context.tolist()
questions = qa_df.question.tolist()
answers = qa_df.answer.tolist()

# Tokenize and encode the text data
input_ids = []
attention_masks = []
start_positions = []
end_positions = []
found_answers = []

for i in range(len(questions)):
    # Tokenize the question and context
    encoded_dict = tokenizer.encode_plus(
        questions[i],
        contexts[i],
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    # Add the encoded results to the lists
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    
    # Find the start and end positions of the answer in the context
    answer_tokens = tokenizer.encode(answers[i], add_special_tokens=False)

    # Search for the first occurrence of the answer sequence in the context
    answer_start_index = None
    answer_end_index = None
    for j in range(len(encoded_dict['input_ids'][0]) - len(answer_tokens) + 1):
        if encoded_dict['input_ids'][0][j:j+len(answer_tokens)].tolist() == answer_tokens:
            answer_start_index = j
            answer_end_index = j + len(answer_tokens) - 1
            break

    # Ensure that the answer was found within the context
    if answer_start_index is None or answer_end_index is None:
        print(f"Could not find the answer in the context for question: {questions[i]}")
        continue

    start_positions.append(answer_start_index)
    end_positions.append(answer_end_index)

    try:
        # Get the index of the first token of the answer in the context
        start_index = encoded_dict.input_ids[0].tolist().index(answer_tokens[0])

        # Get the index of the last token of the answer in the context
        end_index = start_index + len(answer_tokens) - 1

        # Ensure the end index is within the range of the model's limits
        if end_index < 512:
            found_answers.append(i)
            start_positions.append(start_index)
            end_positions.append(end_index)
    except ValueError:
        # If the answer is not found in the tokenized context, it may have been truncated
        print(f"Answer not found in the context for question: {questions[i]}")
        
# Convert the lists into tensors
input_ids = torch.cat([input_ids[i] for i in found_answers], dim=0)
attention_masks = torch.cat([attention_masks[i] for i in found_answers], dim=0)
start_positions = torch.tensor([start_positions[i] for i in found_answers])
end_positions = torch.tensor([end_positions[i] for i in found_answers])

# Create the TensorDataset
dataset = TensorDataset(input_ids, attention_masks, start_positions, end_positions)

# Create the DataLoader
dataloader = DataLoader(
    dataset,
    sampler=RandomSampler(dataset),
    batch_size=8
)

# Set the transformers library to print only error messages
logging.set_verbosity_error()

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
model = BertForQuestionAnswering.from_pretrained('dbmdz/bert-base-turkish-cased')

# Assume 'dataloader' is already defined and loaded with the tokenized dataset
# dataloader = ...

# Define the optimizer using PyTorch's AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

# Define the number of epochs
epochs = 3

# Total number of training steps is the number of batches * number of epochs
total_steps = len(dataloader) * epochs

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Move model to GPU if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
model.train()
for epoch in range(epochs):
    print(f'======== Epoch {epoch + 1} / {epochs} ========')
    total_loss = 0
    
    for step, batch in enumerate(dataloader):
        # Progress update every 10 batches.
        if step % 10 == 0 and not step == 0:
            print(f'Batch {step} of {len(dataloader)}. Loss: {total_loss / step:.2f}')
        
        # Unpack this training batch from our dataloader and copy each tensor to the GPU
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_attention_mask, b_start_positions, b_end_positions = batch
        
        # Clear any previously calculated gradients
        optimizer.zero_grad()
        
        # Perform a forward pass. This will return loss because we provided labels.
        outputs = model(b_input_ids, attention_mask=b_attention_mask, start_positions=b_start_positions, end_positions=b_end_positions)
        
        # Pull the loss value out of the tuple along with the logits because we provided start_positions and end_positions.
        loss = outputs.loss
        total_loss += loss.item()
        
        # Perform a backward pass to calculate gradients
        loss.backward()
        
        # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        
        # Update learning rate schedule
        scheduler.step()
    
    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(dataloader)
    print(f"\nAverage training loss: {avg_train_loss:.2f}")

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_BERT')
tokenizer.save_pretrained('./fine_tuned_BERT')

# Function to ask a question to the fine-tuned model
def ask(question, context):
    # Load the fine-tuned model and tokenizer
    model = BertForQuestionAnswering.from_pretrained('./fine_tuned_BERT')
    tokenizer = BertTokenizer.from_pretrained('./fine_tuned_BERT')

    # Move the model to the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Tokenize the input question and context
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

    # Find the tokens with the highest `start` and `end` scores
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1  # The end index is inclusive for the BERT model, hence +1

    # Convert tokens to the answer string
    answer_tokens = input_ids[0, answer_start:answer_end]
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer_tokens))

    # Remove unwanted tokens and whitespace
    answer = answer.replace('[SEP]', '')
    answer = answer.replace('[CLS]', '')
    answer = ' '.join(answer.split())  # Remove redundant whitespace

    # Additional post-processing could be done here if necessary

    return answer.strip()

# Example usage:
context_example = "Borsa İstanbul Türkiye'nin en büyük borsasıdır ve birçok farklı şirkete ev sahipliği yapmaktadır."
question_example = "Türkiye'nin en büyük borsası hangisidir?"

# Ask the question
answer = ask(question_example, context_example)
print(f"Question: {question_example}")
print(f"Answer: {answer}")
