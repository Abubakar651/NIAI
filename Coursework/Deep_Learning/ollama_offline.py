import ollama

# Define your input dynamically
user_input = "hello"

response = ollama.chat(
    model='qwen2.5:1.5b', 
    messages=[
        {
            'role': 'user',
            'content': user_input, # Your input variable goes here
        },
    ]
)

print(response['message']['content'])
