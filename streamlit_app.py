import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage

# Initialize session state variables if they don't exist
if "history" not in st.session_state:
    st.session_state.history = []
if "current_answer" not in st.session_state: # To store results of the last run temporarily
    st.session_state.current_answer = None
if "current_sources" not in st.session_state:
    st.session_state.current_sources = None
if "current_queries" not in st.session_state:
    st.session_state.current_queries = None
if "current_reflections" not in st.session_state:
    st.session_state.current_reflections = None


st.title("LangGraph Agent Interface")

# Sidebar for configuration
st.sidebar.header("Agent Configuration")
number_of_initial_queries = st.sidebar.number_input("Number of initial search queries", min_value=1, max_value=10, value=3, step=1)
max_research_loops = st.sidebar.number_input("Maximum research loops", min_value=1, max_value=5, value=2, step=1)

# API Key and Question Input Form
with st.form(key="input_form"):
    api_key = st.text_input("Enter your Gemini API Key:", type="password", key="api_key_input")
    user_question = st.text_input("Enter your question:", key="user_question_input")
    submit_button = st.form_submit_button(label="Submit")

# Display conversation history
st.subheader("Conversation History")
if not st.session_state.history:
    st.markdown("*No history yet. Ask a question above!*")

for i, entry in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown("**Answer:**\n" + entry["answer"])
        if entry.get("sources"):
            st.markdown("**Sources:**")
            for source_item in entry["sources"]:
                if isinstance(source_item, dict) and 'url' in source_item:
                    st.markdown(f"- [{source_item.get('title', source_item['url'])}]({source_item['url']})")
                elif isinstance(source_item, str):
                    st.markdown(f"- {source_item}")
                else:
                    st.markdown(f"- {str(source_item)}")
        if entry.get("queries"):
            st.markdown("**Generated Queries:**")
            for q_item in entry["queries"]:
                if isinstance(q_item, dict) and 'query' in q_item:
                    st.markdown(f"- {q_item['query']} (Rationale: {q_item.get('rationale', 'N/A')})")
                elif isinstance(q_item, str):
                    st.markdown(f"- {q_item}")
                else: # Fallback for Query object
                    try:
                        st.markdown(f"- {q_item.query} (Rationale: {getattr(q_item, 'rationale', 'N/A')})")
                    except AttributeError:
                         st.markdown(f"- {str(q_item)}")
        if entry.get("reflections"):
            st.markdown("**Reflection Notes:**")
            for r_idx, note in enumerate(entry["reflections"]):
                st.markdown(f"  *Reflection {r_idx+1}:*")
                st.markdown(f"    - Sufficient: {note.get('is_sufficient', 'N/A')}")
                st.markdown(f"    - Knowledge Gap: {note.get('knowledge_gap', 'N/A')}")
                if 'follow_up_queries' in note and note['follow_up_queries']:
                    st.markdown(f"    - Follow-up Queries: {', '.join(note['follow_up_queries'])}")
    st.divider()


# Processing logic when form is submitted
if submit_button:
    if not api_key:
        st.error("Please enter your Gemini API Key.")
    elif not user_question:
        st.error("Please enter your question.")
    else:
        with st.spinner("Agent is processing your request..."):
            try:
                from backend.src.agent.graph import graph # Import here to use updated API key if changed

                os.environ["GEMINI_API_KEY"] = api_key

                # Pass conversation history to the graph
                # The graph's OverallState expects `messages` to be `Annotated[list, add_messages]`
                # So we need to construct HumanMessage and AIMessage objects
                # For simplicity, we'll just pass the current question for now,
                # as the graph manages its own internal message history based on AIMessage.
                # If the graph was designed to take full chat history as input, we'd map st.session_state.history here.

                invocation_input = {"messages": [HumanMessage(content=user_question)]}
                config = {
                    "configurable": {
                        "number_of_initial_queries": number_of_initial_queries,
                        "max_research_loops": max_research_loops,
                    }
                }

                result = graph.invoke(invocation_input, config=config)

                st.session_state.current_answer = result["messages"][-1].content
                st.session_state.current_sources = result.get("sources_gathered", [])
                st.session_state.current_queries = result.get("generated_queries", [])
                st.session_state.current_reflections = result.get("reflection_notes", [])

                # Append to history
                st.session_state.history.append({
                    "question": user_question,
                    "answer": st.session_state.current_answer,
                    "sources": st.session_state.current_sources,
                    "queries": st.session_state.current_queries,
                    "reflections": st.session_state.current_reflections,
                })

                # Clear current_* from session state as they are now in history
                st.session_state.current_answer = None
                st.session_state.current_sources = None
                st.session_state.current_queries = None
                st.session_state.current_reflections = None

                # Rerun to update the display immediately
                st.rerun()

            except ImportError:
                st.error("Failed to import agent components. Please ensure the backend is correctly set up and all dependencies are installed.")
            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")

# Display areas for the latest response (will be empty if form not submitted or after rerun)
# These are effectively replaced by the history display after successful submission and rerun.
if st.session_state.current_answer: # Should ideally be cleared after adding to history
    st.subheader("Current Answer:")
    st.markdown(st.session_state.current_answer)
if st.session_state.current_sources:
    st.subheader("Current Sources:")
    # (similar display logic as in history)
if st.session_state.current_queries:
    st.subheader("Current Generated Queries:")
    # (similar display logic as in history)
if st.session_state.current_reflections:
    st.subheader("Current Reflection Notes:")
    # (similar display logic as in history)

# Note: The direct display placeholders (answer_placeholder, etc.) are removed
# as the history display and the immediate update via st.rerun() handle this.
# If a non-form based update is needed for placeholders, they could be reintroduced.
# For now, focusing on history.
