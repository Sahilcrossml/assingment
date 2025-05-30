import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from openai import OpenAI
try:
    import chromadb
except ImportError as e:
    st.error(f"Failed to import chromadb. Please ensure all dependencies are installed: {str(e)}")
    st.stop()

from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool
import os
import re

# Confirm frontend is loading
st.write("Streamlit is running! Please enter a URL below.")

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize OpenAI and ChromaDB clients
openai_client = OpenAI()
try:
    chroma_client = chromadb.Client()
except Exception as e:
    st.error(f"Failed to initialize ChromaDB client: {str(e)}")
    st.stop()

def create_embedding(text: str):
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating embedding: {str(e)}")
        return None

def get_collection(name):
    return chroma_client.get_or_create_collection(name=name)

def add_data_to_collection(collection, doc_id, document, embedding):
    if embedding is None:
        return
    # Check if already added to avoid duplicates
    if doc_id not in collection.get()['ids']:
        collection.add(
            documents=[document],
            embeddings=[embedding],
            metadatas=[{"source": doc_id}],
            ids=[doc_id]
        )

def query_relevant_context(collection, user_query: str, top_k=1):
    try:
        query_embedding = create_embedding(user_query)
        if query_embedding is None:
            return ""
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        if results and results['documents'] and len(results['documents'][0]) > 0:
            return results['documents'][0][0]
        else:
            return ""
    except Exception as e:
        st.error(f"Error querying context: {str(e)}")
        return ""

def run_crew(customer, person, inquiry, context):
    Support_agent = Agent(
        role="Senior Support Representative",
        goal="Be the most friendly and helpful support representative in your team.",
        backstory=f"You work at the provided website and are now supporting {customer}.",
        allow_delegation=False,
        verbose=True
    )

    inquiry_task = Task(
        description=(
            f"Answer the following question from {customer}: '{inquiry}' using the provided context:\n{context}"
        ),
        expected_output="A short, helpful response to the customer inquiry.",
        agent=Support_agent
    )

    crew = Crew(
        agents=[Support_agent],
        tasks=[inquiry_task],
        verbose=True,
        memory=True
    )

    try:
        return crew.kickoff(inputs={"customer": customer, "person": person, "inquiry": inquiry, "context": context})
    except Exception as e:
        st.error(f"CrewAI error: {str(e)}")
        return None

st.title("🛠️ Dynamic Support Assistant with URL Input")

# Step 1: User inputs the URL
url = st.text_input("Step 1: Enter the URL to scrape (e.g., https://tailordthreads.myshopify.com/pages/privacy_policy)")

if url:
    # Show a button to confirm URL scraping
    if st.button("Scrape and Embed Content"):
        with st.spinner(f"Scraping {url} and creating embeddings..."):
            try:
                scraper = ScrapeWebsiteTool(website_url=url)
                scraped_text = scraper.run()
                if not scraped_text or "404 Page not found" in scraped_text:
                    st.error("The provided URL returned a 404 Page Not Found error. Please check the URL (e.g., use 'privacy_policy' instead of 'privacy-policy').")
                else:
                    st.success(f"Scraped {len(scraped_text)} characters from the page!")
                    st.write("Debug: First 500 characters of scraped text:", scraped_text[:500])
                    embedding = create_embedding(scraped_text)
                    if embedding:
                        # Enhanced sanitization: replace non-alphanumeric characters (except underscores and hyphens) with underscores
                        # Ensure starts and ends with alphanumeric
                        raw_name = re.sub(r'[^a-zA-Z0-9_-]', '_', url.replace("https://", ""))
                        collection_name = f"c_{raw_name[:48]}" if raw_name[0] in ['_', '-'] else raw_name[:50]
                        if not collection_name[-1].isalnum():
                            collection_name = collection_name[:-1] + 'c'
                        if len(collection_name) < 3:
                            collection_name = collection_name + 'c' * (3 - len(collection_name))
                        collection = get_collection(collection_name)
                        add_data_to_collection(collection, doc_id="page_1", document=scraped_text, embedding=embedding)
                        st.session_state['collection_name'] = collection_name
                        st.session_state['scraped_text'] = scraped_text
                        st.write("Debug: Collection name:", collection_name)
                    else:
                        st.error("Failed to create embedding for scraped text.")
            except Exception as e:
                st.error(f"Error scraping or embedding: {str(e)}")

if 'collection_name' in st.session_state:
    st.write("### Step 2: Ask your question based on the scraped content")

    customer_name = st.text_input("Customer Name", "Sahil Dhiman", key="cust_name")
    user_question = st.text_area("Your Question")

    if st.button("Get Answer"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating response..."):
                try:
                    collection = get_collection(st.session_state['collection_name'])
                    context = query_relevant_context(collection, user_query=user_question)
                    st.write("Debug: Retrieved Context (first 500 characters):", context[:500] if context else "No context retrieved")
                    result = run_crew(customer_name, "bot", user_question, context)
                    if result:
                        st.write("#### Response")
                        st.markdown(result.tasks_output[0].raw)  # Display only the support response as formatted text
                    else:
                        st.error("Failed to generate a response.")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
