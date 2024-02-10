from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import os

app = Flask(__name__)

# Extract the contents of the zip file
with zipfile.ZipFile('dawnnews.zip', 'r') as zip_ref:
    zip_ref.extractall('dawnnews')

# List the files in the extracted folder
extracted_files = os.listdir('dawnnews')

# Accumulate documents from all CSV files
all_documents = []

for csv_file_name in [file for file in extracted_files if file.endswith('.csv')]:
    csv_file_path = os.path.join('dawnnews', csv_file_name)

    try:
        # Load the dataset with dtype=str to handle mixed types
        df = pd.read_csv(csv_file_path, dtype=str)

        # Assuming the text is in the second column, update the index accordingly
        documents = df.iloc[:, 1].values  # Use the correct column index or name

        # Replace NaN values with an empty string
        documents = ["" if pd.isna(doc) else doc for doc in documents]

        all_documents.extend(documents)
    except pd.errors.ParserError as e:
        print("Error reading file {}: {}".format(csv_file_name, e))

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(all_documents)

# Function to get relevant documents based on user query
# Function to get relevant documents based on user query
def get_relevant_documents_with_details(query, num_results=5):
    # Transform the query to a TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity between the query vector and document vectors
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Get indices of top N most similar documents
    top_indices = similarity_scores.argsort()[0][::-1][:num_results]

    # Display relevant documents with details
    relevant_documents = []

    print(f"Top Indices: {top_indices}")

    for idx in top_indices:
        try:
            title = all_documents[idx]
            description = documents[idx]
            relevant_documents.append({"title": title, "description": description})
        except IndexError as e:
            print(f"Error retrieving document at index {idx}: {e}")
            continue  # Skip to the next index if an error occurs

    print(f"Relevant Documents: {relevant_documents}")

    return relevant_documents




@app.route('/details/<int:document_index>')
def document_details(document_index):
    document_info = all_documents[document_index]
    return render_template('details.html', document_info=document_info)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        results = get_relevant_documents_with_details(query, num_results=8)
        return render_template('index.html', query=query, results=results)
    else:
        return render_template('index.html', query=None, results=None)

if __name__ == '__main__':
    app.run(debug=True)
