import requests
import json

def scrape_shl_product_catalog():
    # The URL containing the JSON data
    url = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
    output_filename = "catalog_raw.json"

    print(f"Fetching data from {url}...")
    
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()
        
        # Read the raw text from the response
        raw_text = response.text
        
        # Parse the JSON with strict=False to allow unescaped control characters
        data = json.loads(raw_text, strict=False)
        
        # Save the JSON data into a file
        with open(output_filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
            
        print(f"Successfully scraped the data and saved it to '{output_filename}'.")
        print(f"Total records saved: {len(data)}")

    except json.JSONDecodeError as json_err:
        print(f"JSON Decoding Error: {json_err}")
        print("The JSON on the server is heavily malformed. Saving raw text to a fallback file...")
        
        # Fallback: Save the raw text so you don't lose the data
        fallback_filename = "shl_product_catalog_raw_fallback.txt"
        with open(fallback_filename, 'w', encoding='utf-8') as raw_file:
            raw_file.write(response.text)
        print(f"Saved the raw, unparsed data to '{fallback_filename}'. You can inspect line {json_err.lineno} manually.")
        
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred while fetching the data: {req_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_distinct_keys(filename):
    # Use a set to automatically keep only unique values
    distinct_keys = set()
    
    print(f"Reading data from '{filename}'...\n")
    
    try:
        # Open and load the JSON file
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # Iterate through each product in the JSON array
        for item in data:
            # Check if the 'keys' field exists and is a list
            if "keys" in item and isinstance(item["keys"], list):
                # Add each key to our set
                for key in item["keys"]:
                    # Stripping whitespace just in case there are formatting inconsistencies
                    distinct_keys.add(key.strip())
                    
        # Convert the set to a sorted list for better readability
        sorted_keys = sorted(list(distinct_keys))
        
        # Print the results
        print(f"Found {len(sorted_keys)} distinct keys:")
        print("-" * 40)
        for key in sorted_keys:
            print(f"- {key}")
            
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found. Make sure you are in the correct directory.")
    except json.JSONDecodeError as json_err:
        print(f"Error: Could not parse JSON from the file. {json_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # scrape_shl_product_catalog()
    target_file = "catalog_raw.json"
    get_distinct_keys(target_file)