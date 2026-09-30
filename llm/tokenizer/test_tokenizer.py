from tokenizer import SimpleTokenizer


text = """
CPU scheduling is the process of selecting
a process from the ready queue.
CPU scheduling is important.
"""

tokenizer = SimpleTokenizer()

tokenizer.build_vocabulary(text)

print("Vocabulary:")
print(tokenizer.token_to_id)

sentence = "CPU scheduling is important."

encoded = tokenizer.encode(sentence)

print("\nEncoded:")
print(encoded)

decoded = tokenizer.decode(encoded)

print("\nDecoded:")
print(decoded)