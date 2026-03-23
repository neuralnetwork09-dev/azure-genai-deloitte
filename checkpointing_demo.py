import os  # Used for environment variables (not directly used here but commonly needed)
from typing import TypedDict, List  # Helps define structured data types
from dotenv import load_dotenv  # Loads variables from .env file
from langgraph.graph import StateGraph, END  # Core LangGraph components
from langgraph.checkpoint.memory import MemorySaver  # Stores data in memory

# Load environment variables from .env file
load_dotenv()


# Define the structure of our state (data passed between steps)
class ConversationState(TypedDict):
    messages: List[str]   # List to store messages
    iteration: int        # Counter to track how many times loop ran


# Function that runs in each step of the graph
def add_message(state: ConversationState) -> dict:
    # Increase iteration count by 1
    iteration = state.get("iteration", 0) + 1

    # Get existing messages (or empty list if none)
    messages = state.get("messages", [])

    # Add a new message for this step
    messages.append(f"Step {iteration} completed")

    # Print progress to console
    print(f"[add_message] Iteration {iteration}")

    # Return updated state
    return {"messages": messages, "iteration": iteration}


# Function to decide whether to continue or stop
def should_continue(state: ConversationState) -> str:
    # If we have completed 3 iterations, stop the graph
    if state["iteration"] >= 3:
        return END

    # Otherwise, continue running add_message again
    return "add_message"


# Create a graph with our state structure
graph = StateGraph(ConversationState)

# Add a node (step) called "add_message"
graph.add_node("add_message", add_message)

# Define where the graph should start
graph.set_entry_point("add_message")

# Define conditional flow (loop or stop)
graph.add_conditional_edges("add_message", should_continue)


# Use MemorySaver to store progress in memory (no database or file needed)
checkpointer = MemorySaver()

# Compile the graph into a runnable app
app = graph.compile(checkpointer=checkpointer)


# Configuration (used to identify this run / session)
config = {"configurable": {"thread_id": "demo-thread-001"}}


# Start execution
print("=== Checkpointing Demo ===")

# Initial input: empty messages and iteration = 0
result = app.invoke({"messages": [], "iteration": 0}, config)

# Print final result
print(f"Final messages: {result['messages']}")


# Show history of all steps (checkpointed states)
print("\nState history:")
for snapshot in app.get_state_history(config):
    print(
        f"  Iteration: {snapshot.values.get('iteration', 0)}  "
        f"Messages: {len(snapshot.values.get('messages', []))}"
    )