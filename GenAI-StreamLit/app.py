# app.py -This is the Streamlit front end
import os
import streamlit as st
from openai import OpenAI
from utils import build_messages_from_template, call_finetuned_model, call_base_model

st.set_page_config(page_title="Car Dealer AI (Fine-Tuned)", layout="centered")

st.title("Car Dealer AI — Fine-Tuned Model Demo")
st.write("Demo app that uses your fine-tuned OpenAI model for car-dealer prompts.")

# Sidebar config
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("OPENAI API Key", type="password", value=os.getenv("OPENAI_API_KEY"))
ft_model = st.sidebar.text_input("Fine-tuned model name (ft model id)", value=os.getenv("FT_MODEL_NAME"))
use_finetuned = st.sidebar.checkbox("Use fine-tuned model (if available)", value=True)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0)
max_tokens = st.sidebar.number_input("Max tokens", min_value=64, max_value=2048, value=400, step=64)
stream_logs = st.sidebar.checkbox("Show raw API response", value=False)

if not api_key:
    st.warning("Enter your OpenAI API key in the sidebar to call the API.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)


# THis will provide a dropdown in UI so that user can select different templates
st.header("Choose a prompt template")
template = st.selectbox("Template", [
    "Personalized Sales Email (SUV)",
    "Social Media Ads (Electric Car)",
    "Upsell Add-ons (Sedan)",
    "Customer Chat Simulation (Hybrid vs Gas)",
    "Limited-time Trade-in Promo (SUV)",
    "Custom Prompt (raw input)"
])

st.header("Prompt inputs")
if template == "Personalized Sales Email (SUV)":
    # User car change the default values
    customer_name = st.text_input("Customer name", "Alex")
    tone = st.selectbox("Tone", ["Friendly","Professional","Urgent"])
    user_prompt = f"Write a personalized sales email to {customer_name} who is interested in SUVs. Tone: {tone}. Highlight safety, space, and fuel efficiency. End with a CTA for a test drive."
elif template == "Social Media Ads (Electric Car)":
    model_name = st.text_input("Electric model name", "Electric X1")
    user_prompt = f"Generate three Facebook ad variations promoting the {model_name}. Target: young professionals, families, eco-conscious buyers."
elif template == "Upsell Add-ons (Sedan)":
    user_prompt = "Suggest three add-ons to offer to a customer who just bought a sedan. Use a friendly sales tone."
elif template == "Customer Chat Simulation (Hybrid vs Gas)":
    user_prompt = "Act as a virtual car dealer assistant and persuasively explain why a hybrid car is better than a traditional gas car."
elif template == "Limited-time Trade-in Promo (SUV)":
    deadline = st.text_input("Offer end date (e.g., Dec 1, 2025)", "Dec 1, 2025")
    user_prompt = f"Create a professional and urgent promotional message offering a limited-time discount for a trade-in deal on SUVs that ends on {deadline}."
else:
    user_prompt = st.text_area("Custom prompt", value="Write a friendly short ad for a compact car.")

st.write("### Final prompt to model")
st.code(user_prompt, language="text")

if st.button("Generate"):
    with st.spinner("Calling OpenAI..."):
        messages = build_messages_from_template(user_prompt)
        try:
            if use_finetuned and ft_model:
                resp = call_finetuned_model(client, ft_model, messages, temperature, max_tokens)
            else:
                resp = call_base_model(client, messages, temperature, max_tokens)
            st.subheader("Model output")
            st.write(resp)
            if stream_logs:
                st.markdown("**Raw response object**")
                # The helper returns string; if you want raw response, modify helper to return response
                st.info("Enable raw API response in utils if desired.")
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

st.markdown("---")
st.caption("Notes: Use `FT_MODEL_NAME` env var or set the fine-tuned model id in the sidebar. Fine-tuned model IDs look like `ft:...`.")
