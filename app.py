import gradio as gr
from transformers import pipeline, AutoTokenizer
from personal_info import PERSONAL_CONTEXT

# ────────────────────────────────────────────────
#   Smallest stable model + fastest settings
# ────────────────────────────────────────────────

model_name = "Qwen/Qwen2-0.5B-Instruct"

print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    model_kwargs={
        "low_cpu_mem_usage": True,
    }
)

print("Model loaded.")

def chat_with_pathum(message, history):
    # Build messages – keep very short context
    messages = [{"role": "system", "content": PERSONAL_CONTEXT}]

    # Only last 3 full turns to save memory
    for user_msg, assistant_msg in history[-3:]:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    # Show immediate feedback
    yield "Thinking... (usually 8–50 seconds on free CPU)"

    try:
        outputs = pipe(
            messages,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.75,
            top_p=0.92,
            repetition_penalty=1.15,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )

        reply = outputs[0]["generated_text"].strip()

        # Clean minimal artifacts
        if reply.startswith("assistant:"):
            reply = reply.replace("assistant:", "", 1).strip()

        yield reply

    except Exception as e:
        yield f"Oops... something went wrong: {str(e)[:100]}"

# ────────────────────────────────────────────────
#             Gradio Interface
# ────────────────────────────────────────────────

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🤖 Chat with Pathum (Free CPU Version)

        Ask anything about me!  
        → First message takes longest (model loading)  
        → Later messages usually 8–50 seconds  
        → Free CPU is limited — thanks for your patience! 😊
        """
    )

    gr.ChatInterface(
        fn=chat_with_pathum,
        examples=[
            "Who are you?",
            "Where are you from?",
            "What do you do?",
            "Tell me about your projects",
            "ඔයා මොකද්ද කරන්නේ මචන්? 😄"
        ],
        title="Pathum's Personal Chatbot",
        description="Qwen2-0.5B-Instruct • Built by Pasindu Pathum",
        autofocus=True,
    )

demo.launch(
    server_name="0.0.0.0",
    theme=gr.themes.Soft()
)
