# BOOLEAN IR MODEL

## Overview
This project implements a Boolean query using the **Shunting-Yard Algorithm** to convert infix Boolean expressions into postfix (Reverse Polish Notation) and evaluate them efficiently. The processor supports logical operators:
- `AND` (higher precedence than OR)
- `OR`
- `NOT` (highest precedence)
- Parentheses for grouping

## Requirements
### Dependencies
Make sure you have Python installed. You also need the following libraries:

```bash
pip install nltk
```

## How to Run the Code
### 1. Prepare Your Data
Ensure that the `archive/CISI.ALL` file is present in the working directory. This file contains documents that will be indexed for Boolean retrieval.

### 2. Run the Script
Execute the script using:

```bash
python3 project.py
```

This will:
1. **Read and parse documents** from the `CISI.ALL` file.
2. **Preprocess the queries** by tokenizing them and converting them into postfix notation using the Shunting-Yard algorithm.
3. **Evaluate the queries** against the document index and return relevant document IDs.

## What Happens When You Run It?
1. **Document Parsing:**
   - The `read_documents()` function reads the `CISI.ALL` file, extracts document IDs and content, and stores them in a dictionary.

2. **Tokenization & Query Processing:**
   - The `shunting_yard(tokens)` function converts infix Boolean expressions into postfix notation, ensuring correct precedence.
   - Example:
     ```
     (A AND B) OR C  →  A B AND C OR
     ```

3. **Boolean Query Execution:**
   - This will execute some boolean queries to show how it works.
   - Unfortunately I do not provide a method to evaluate the results of the boolean queries since I do not have the gold results.


4. **Phrase Query Execution:**
   - Here I test the gold queries and (by using Cosine similarity) I check which documents are more relevant than others.

5. **Precision-Recall Analysis (Optional):**
   - If relevance judgments are available, the system can compute precision, recall, and plot precision-recall curves.


```

## Additional Features
- **Precision-Recall Curve**: Evaluate retrieval effectiveness by computing precision and recall at different thresholds.
- **Mean Average Precision (MAP)**: Evaluate overall ranking performance.

## Notes
- The document collection and queries should be formatted properly.
- The algorithm assumes Boolean queries are correctly formatted with valid parentheses.
- If queries contain mismatched parentheses, an error is raised.

## Author
- Developed by Giacomazzi Andrea

## License
- MIT License


