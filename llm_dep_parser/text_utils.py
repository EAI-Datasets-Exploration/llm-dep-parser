import json
import pandas as pd
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity


# Function to Construct Prompts
def format_prompt(text):
    return f"""
    Perform dependency parsing on the following robotics command:

    Sentence: "{text}"

    Provide the output in a **valid JSON format** with the following structure:

    ```json
    {{
      "sentence": "PICK UP the red block",
      "tokens": [
        {{"text": "PICK", "lemma": "pick", "pos": "VERB", "head": 1, "dep": "ROOT"}},
        {{"text": "UP", "lemma": "up", "pos": "ADP", "head": 0, "dep": "prt"}},
        {{"text": "the", "lemma": "the", "pos": "DET", "head": 4, "dep": "det"}},
        {{"text": "red", "lemma": "red", "pos": "ADJ", "head": 4, "dep": "amod"}},
        {{"text": "block", "lemma": "block", "pos": "NOUN", "head": 1, "dep": "dobj"}}
      ]
    }}
    ```

    **Token Fields Explanation:**
    - `"text"`: The original word in the sentence.
    - `"lemma"`: The base (dictionary) form of the word.
    - `"pos"`: Part of Speech (e.g., VERB, NOUN, ADJ, etc.).
    - `"head"`: The index of the word that this token is dependent on.
    - `"dep"`: The dependency relation label (e.g., `ROOT`, `dobj`, `amod`, etc.).

    Ensure the output is in **valid JSON format** with proper nesting and data types.
    """


##########################
### All TFIDF Features ###
##########################


def make_label_df(path_to_labels_df):
    df = pd.read_csv(path_to_labels_df)
    df = df[df["annotation_quality"].isin(["good", "bad"])]

    def extract_text_example(prompt):
        match = re.search(r'Sentence:\s*"([^"]+)"', prompt)
        return match.group(1) if match else None

    df["text_example"] = df["full_prompt"].apply(extract_text_example)
    return df


def load_tfidf(vectorizer_fp, matrix_fp):
    with open(vectorizer_fp, "rb") as f:
        loaded_vectorizer = pickle.load(f)
    with open(matrix_fp, "rb") as f:
        loaded_matrix = pickle.load(f)
    return loaded_vectorizer, loaded_matrix


def find_closest_texts(query_vec, tfidf_matrix, path_to_labels_df, top_n=3):
    # Find the closest text examples to the given query using precomputed TF-IDF.
    similarities = cosine_similarity(
        query_vec, tfidf_matrix
    ).flatten()  # Compute similarities
    top_indices = similarities.argsort()[-top_n:][::-1]  # Get top matches
    df = make_label_df(path_to_labels_df)
    return df.iloc[top_indices]["text_example"].tolist()


def get_label(text, path_to_labels_df):
    def get_relevant_columns(df, search_text):
        """
        Search for a specific text in 'text_example' column and 
        retrieve relevant columns based on 'annotation_quality' value.
        """
        # Find the index where 'text_example' matches the search text
        index = df[df["text_example"] == search_text].index

        if len(index) == 0:
            return "Text not found in 'text_example' column."

        index = index[0]  # Get the first matching index

        # Check the annotation quality
        annotation_quality = df.at[index, "annotation_quality"]

        # Select columns based on annotation quality
        if annotation_quality == "good":
            return {
                "example_sentence": text,
                "direct_objects": df.at[index, "direct_objects"],
                "verbs": df.at[index, "verbs"],
            }
        elif annotation_quality == "bad":
            return {
                "example_sentence": text,
                "direct_objects": df.at[index, "corrected_direct_objects"],
                "verbs": df.at[index, "corrected_verbs"],
            }
        else:
            return ""

    df = make_label_df(path_to_labels_df)
    return get_relevant_columns(df, text)


def format_prompt_icl(text, vectorizer_fp, matrix_fp, path_to_labels_df):
    """
    Generate a structured prompt for extracting direct objects and verbs
    while incorporating in-context learning (ICL) examples.
    """
    # Load precomputed TF-IDF model
    vectorizer, tfidf_matrix = load_tfidf(vectorizer_fp, matrix_fp)

    # Find the closest text examples using similarity search
    query_vector = vectorizer.transform([text])
    closest_texts = find_closest_texts(query_vector, tfidf_matrix, path_to_labels_df)

    # Retrieve ICL examples based on closest text matches
    icl_examples = [get_label(ex, path_to_labels_df) for ex in closest_texts]

    # Format in-context learning examples
    icl_string = "\n".join(
        f"""Example {i+1}:
    Sentence: "{ex['example_sentence']}"
    Output:
    {{
        "direct_objects": {ex['direct_objects']},
        "verbs": {ex['verbs']}
    }}\n"""
        for i, ex in enumerate(icl_examples)
        if ex  # Ensuring valid examples are included
    )

    # Construct the final prompt with ICL examples
    return f"""
    Extract the direct objects and verbs from the following sentence while considering prepositional phrases. 
    Follow these steps:
    
    1. Identify the verb(s) in the sentence.
       - Look for the main action or state of being.
       - If there is a verb phrase (e.g., "has been running"), include the full phrase.
    
    2. Identify the subject by asking:
       - "Who?" or "What?" before the verb.
    
    3. Locate and temporarily ignore any prepositional phrases:
       - Identify phrases that start with prepositions ("to," "in," "on," "at," "for," "with," "about," "by," "over," "under," etc.).
       - Words within these phrases should not be considered direct objects.
    
    4. Find the direct object by asking:
       - "What?" or "Whom?" after the verb.
       - Ensure the answer is NOT inside a prepositional phrase.
    
    5. Cross-check the sentence:
       - If removing prepositional phrases leaves a meaningful sentence with a noun receiving the action, that noun is the direct object.
       - If no noun answers "What?" or "Whom?" after the verb, the sentence may not have a direct object.
    
    6. Confirm by distinguishing between action and linking verbs:
       - If the verb is a linking verb ("is," "are," "was," "were," "be," etc.), there is no direct object—only a subject complement.

    ### **Examples for Reference:**
    {icl_string}
    
    Now, extract the direct objects and verbs from the following sentence and return them in JSON format:

    Sentence: "{text}"

    Output format:
    {{
        "direct_objects": ["object1", "object2", ...],
        "verbs": ["verb1", "verb2", ...]
    }}
    """


##########################
### End TFIDF Features ###
##########################


# Extract JSON from LLM Response
def extract_json(text):
    """
    Extracts and parses the final JSON block from the LLM response.

    Returns:
        direct_objects (list): List of extracted direct objects.
        verbs (list): List of extracted verbs.
    """
    # Find all JSON occurrences in the response
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)

    if matches:
        # Use only the last detected JSON block (most likely the final answer)
        json_text = matches[-1]

        # Clean up formatting issues
        json_text = json_text.replace("```json", "").replace(
            "```", ""
        )  # Remove Markdown code block
        json_text = json_text.replace("“", '"').replace("”", '"')  # Fix curly quotes
        json_text = json_text.replace("\n", "").strip()  # Remove newlines and spaces

        # Check for Placeholder JSON Output (detect and skip)
        if "object1" in json_text or "verb1" in json_text or "..." in json_text:
            print(f"Skipping template JSON: {json_text}")
            return [], []

        # Attempt to parse JSON
        try:
            extracted_json = json.loads(json_text)
            direct_objects = extracted_json.get("direct_objects", [])
            verbs = extracted_json.get("verbs", [])

            # Ensure lists (handle cases where model outputs a string instead of a list)
            direct_objects = (
                direct_objects if isinstance(direct_objects, list) else [direct_objects]
            )
            verbs = verbs if isinstance(verbs, list) else [verbs]

            return direct_objects, verbs

        except json.JSONDecodeError:
            print(f"JSON decoding error: {json_text}")  # Debugging info
            return [], []  # Return empty lists if JSON parsing fails

    return [], []  # Return empty lists if no JSON is found


# Function to remove determiners for sorting
def clean_determiners(text):
    return re.sub(
        r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE
    )  # Remove "the", "a", "an" at the start
