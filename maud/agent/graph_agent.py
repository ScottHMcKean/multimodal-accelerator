from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END
from state import load_graph_state
from nodes import (contains_chat_history, 
                   query_vector_database,
                   generation_with_history, 
                   load_generation_no_history)

state = load_graph_state()
generation_no_history = load_generation_no_history()

def load_graph() -> CompiledStateGraph:
  workflow = StateGraph(state)
  workflow.add_node("query_vector_database", query_vector_database)
  workflow.add_node("generation_with_history", generation_with_history)
  workflow.add_node("generation_no_history", generation_no_history)

  workflow.set_conditional_entry_point(contains_chat_history,
                                      {"contains_history": "generation_with_history",
                                        "no_history": "query_vector_database"})

  workflow.add_edge("generation_with_history", "query_vector_database")
  workflow.add_edge("query_vector_database", "generation_no_history")
  workflow.add_edge("generation_no_history", END)

  return workflow.compile()
