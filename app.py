import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from openai import OpenAI
import chromadb
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool
import os

# Confirm frontend is loading
st.write("Streamlit is running! Please enter a URL below.")

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize OpenAI and ChromaDB clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.Client()

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

    QA_agent = Agent(
        role="Support Quality Assurance Specialist",
        goal="Ensure support quality is excellent.",
        backstory=f"You are reviewing the support response for {customer}.",
        verbose=True
    )

    inquiry_task = Task(
        description=(
            f"{customer} just asked: '{inquiry}'\n\nUse this context:\n{context}"
        ),
        expected_output="A short, helpful response to the customer inquiry.",
        agent=Support_agent
    )

    qa_task = Task(
        description=(
            f"Review the response given for the inquiry: '{inquiry}'"
        ),
        expected_output="Detailed QA review of the response, confirming its accuracy and tone.",
        agent=QA_agent
    )

    crew = Crew(
        agents=[Support_agent, QA_agent],
        tasks=[inquiry_task, qa_task],
        verbose=True,
        memory=True
    )

    return crew.kickoff(inputs={"customer": customer, "person": person, "inquiry": inquiry})

st.title("🛠️ Dynamic Support Assistant with URL Input")

# Step 1: User inputs the URL
url = st.text_input("Step 1: Enter the URL to scrape (e.g., https://tailordthreads.myshopify.com/collections/all)")

if url:
    # Show a button to confirm URL scraping
    if st.button("Scrape and Embed Content"):
        with st.spinner(f"Scraping {url} and creating embeddings..."):
            try:
                scraper = ScrapeWebsiteTool(website_url=url)
                scraped_text = scraper.run()
                if not scraped_text:
                    st.error("Failed to scrape content from the provided URL.")
                else:
                    st.success(f"Scraped {len(scraped_text)} characters from the page!")
                    st.write("Debug: First 500 characters of scraped text:", scraped_text[:500])
                    embedding = create_embedding(scraped_text)
                    if embedding:
                        collection_name = url.replace("https://", "").replace("/", "_")[:50]  # Sanitize name
                        collection = get_collection(collection_name)
                        add_data_to_collection(collection, doc_id="page_1", document=scraped_text, embedding=embedding)
                        st.session_state['collection_name'] = collection_name
                        st.session_state['scraped_text'] = scraped_text
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
                    context = query_relevant_context(collection, user_question)
                    st.write("Debug: Retrieved Context (first 500 characters):", context[:500] if context else "No context retrieved")
                    result = run_crew(customer_name, "bot", user_question, context)
                    st.write("#### Support Assistant Response")
                    st.markdown(result.tasks_output[0].raw)  # Display support response as formatted text
                    st.write("#### QA Review")
                    st.markdown(result.tasks_output[1].raw)  # Display QA review as formatted text
                    st.write("#### Full JSON Response (for debugging)")
                    st.json(result)  # Keep JSON for reference
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")