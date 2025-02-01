import openai
import csv
import os
from datasets import load_dataset

# Load dataset (Python code snippets from CodeSearchNet)
dataset = load_dataset("code_search_net", "python", trust_remote_code=True)

# Secure OpenAI API Key using Environment Variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("ERROR: OpenAI API Key is missing. Set OPENAI_API_KEY as an environment variable.")
client = openai.OpenAI(api_key=api_key)

# Function to request readability labels from GPT-4
def get_readability_score(code_snippet):
    print(f"DEBUG: Sending to GPT-4 -> {code_snippet[:100]}...")  
    prompt = f"""
    Rate the readability of the following Python function from 1 to 5 (1 = unreadable, 5 = perfect readability).
    Also, provide a short reason:
    
    Code:
    {code_snippet}
    
    Answer format: [Score] - [Reason]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant trained to assess code readability."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        print(f"DEBUG: GPT-4 Response -> {response}")  
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"ERROR: GPT-4 request failed: {e}")
        return "Error - Failed to generate response"


# Process dataset and save labeled data
output_csv = "src/data/readability_dataset.csv"
with open(output_csv, "w", newline="") as csvfile:
    fieldnames = ["code_snippet", "readability_score", "reason"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i, sample in enumerate(dataset["train"][:1000]):  # 1000 samples
        if isinstance(sample, dict):  
            code_snippet = sample.get("func_code_string")  
            if not code_snippet:
                print(f"Skipping sample {i+1}")
                continue  

            try:
                label = get_readability_score(code_snippet)  
                print(f"DEBUG: GPT-4 Response -> {label}")
                score, reason = label.split(" - ", 1)  
                writer.writerow({"code_snippet": code_snippet, "readability_score": score, "reason": reason})
                print(f"Processed {i+1}/1000")
            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")

print(f"Readability dataset saved as {output_csv}")
