import os

from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variables for model and tokenizer
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Initialize chat history with a system message
chat_history = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
]

def generate_response(user_input, max_new_tokens=1024, max_history_length=5):
    """
    Generates a response using a preloaded model and tokenizer, maintaining a chat history.

    Args:
        user_input (str): The user's input message.
        max_new_tokens (int): The maximum number of tokens to generate. Default is 512.
        max_history_length (int): The maximum number of messages to retain in the chat history. Default is 5.

    Returns:
        str: The generated response from the model.
    """
    global chat_history

    # Add the new user message to the chat history
    chat_history.append({"role": "user", "content": user_input})

    # Ensure the chat history does not exceed the maximum length
    if len(chat_history) > max_history_length:
        chat_history.pop(1)  # Remove the second message (first user message) to maintain context

    # Tokenize the messages using the chat template
    text = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate the response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the generated response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Add the model's response to the chat history
    chat_history.append({"role": "assistant", "content": response})

    return response

# Example usage
user_input = "Give me a short introduction to large language models."
response = generate_response(user_input)
print(response)

print(os.run("nvidia-smi"))

# Subsequent interactions
user_input = "Can you explain how they are trained?"
response = generate_response(user_input)
print(response)
