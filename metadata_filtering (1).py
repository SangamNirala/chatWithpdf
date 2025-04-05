import json

# ✅ Function to apply metadata filtering
def filter_documents_by_metadata(documents, metadata_filters):
    """
    Filters documents based on metadata criteria.
    
    :param documents: List of documents with metadata.
    :param metadata_filters: Dictionary of filters (e.g., {"author": "John Doe"}).
    :return: Filtered list of documents.
    """
    filtered_docs = []
    for doc in documents:
        match = all(doc.metadata.get(key) == value for key, value in metadata_filters.items())
        if match:
            filtered_docs.append(doc)
    
    return filtered_docs

# ✅ Function to load metadata from JSON
def load_metadata(metadata_file):
    """
    Loads metadata from a JSON file.
    
    :param metadata_file: Path to JSON file.
    :return: Dictionary of metadata.
    """
    try:
        with open(metadata_file, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"⚠️ Error loading metadata: {e}")
        return {}
