import gradio as gr

def create_interface(process_query, agent_manager, session_id):
    session_id = session_id
    print("interface.py session_id :", session_id)
    with gr.Blocks(title="AI Assistant") as demo:
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
            height=400
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                scale=8
            )
            submit = gr.Button("Submit", scale=1)
        
        clear = gr.Button("Clear")

        def process_message(message, history):
            response = process_query(message, history, session_id)
            history.append((message, response))
            return "", history


        def clear_session():
            return agent_manager.clear_context(session_id)

        
        msg.submit(
            process_message,
            [msg, chatbot],
            [msg, chatbot]
        )
        
        submit.click(
            process_message,
            [msg, chatbot],
            [msg, chatbot]
        )
        
        clear.click(
            clear_session,
            None,
            [chatbot, msg]
        )

    return demo