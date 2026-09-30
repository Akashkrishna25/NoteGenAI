def create_notes_prompt(
    topic,
    level="beginner",
    length="medium"
):

    prompt = f"""
You are an AI study assistant.

Generate study notes about:

Topic: {topic}
Difficulty: {level}
Length: {length}

Structure the notes using the following sections:

1. Introduction
2. Key Concepts
3. Detailed Explanation
4. Real-World Example
5. Advantages
6. Disadvantages
7. Important Points
8. Interview Questions
9. Quick Revision

Make the explanation clear and educational.

Use simple language for beginners.

Topic:
{topic}
"""

    return prompt