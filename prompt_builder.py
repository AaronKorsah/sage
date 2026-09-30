def build_prompt(question, chunks):
    """
    Takes a question and a list of chunks.
    Returns a single formatted prompt string.
    """
    context = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""

    return prompt


if __name__ == "__main__":
    question = "How many days of annual leave do employees get?"

    chunks = [
        "Full-time employees are entitled to 25 days of annual leave per year. Sick leave requires a medical certificate for absences over 2 days.",
        "Salaries are paid on the 28th of each month. Bonuses are paid quarterly based on performance targets."
    ]

    prompt = build_prompt(question, chunks)

    print(prompt)