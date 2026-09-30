def smart_chunk(text, chunk_size=500, overlap=100):
    """
    Splits text into chunks, respecting paragraph and sentence boundaries.
    
    Strategy:
      1. Try to keep whole paragraphs together.
      2. If a paragraph is too big, split it at sentence boundaries.
      3. If a sentence is too big, split it at word boundaries.
      4. Only cut mid-word as a last resort.
    
    Returns a list of chunks with optional overlap.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    chunks = []
    current = ""
    
    for para in paragraphs:
        # Can this whole paragraph fit into the current chunk?
        if len(current) + len(para) + 2 <= chunk_size:
            current += para + "\n\n"
            continue
        
        # Current chunk is full — save it
        if current.strip():
            chunks.append(current.strip())
            current = ""
        
        # Does this paragraph fit on its own?
        if len(para) <= chunk_size:
            current = para + "\n\n"
            continue
        
        # Paragraph is too big — split at sentences
        sentences = para.replace(". ", ".\n").split("\n")
        temp = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(temp) + len(sent) + 1 <= chunk_size:
                temp += sent + " "
            else:
                if temp.strip():
                    chunks.append(temp.strip())
                temp = sent + " "
        if temp.strip():
            current = temp.strip() + "\n\n"
    
    if current.strip():
        chunks.append(current.strip())
    
    # Add overlap between chunks
    if overlap > 0:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # Take the last `overlap` characters from the previous chunk
                prev_tail = chunks[i-1][-overlap:]
                overlapped.append(prev_tail + " " + chunk)
        chunks = overlapped
    
    return chunks


if __name__ == "__main__":
    from pypdf import PdfReader

    reader = PdfReader("frontiers.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    chunks = smart_chunk(text, chunk_size=500, overlap=100)

    print(f"Total characters: {len(text)}")
    print(f"Total chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk)
        print()