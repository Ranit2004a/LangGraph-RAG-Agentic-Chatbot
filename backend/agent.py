from typing import TypedDict, List, Literal
from langchain_core.messages import  BaseMessage,HumanMessage,AIMessage,SystemMessage
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq
import os
from config import GROQ_API_KEY,TAVIL_API_KEY, PINECONE_API_KEY 


#pydentic schema for strctured output 
class RouteDecision(BaseModel):
    route: Literal["rag", "web", "answer", "end"]
    reply: str | None = Field(None, description="Filled only when router  == 'end'")
    
class RagJudge(BaseModel):
    sufficient : bool = Field (..., description="True if retrieved documents are sufficient to answer the question, False otherwise")

#LLM Agent state schema
os.environ["GROQ_API_KEY"] =GROQ_API_KEY  

router_llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY, temperature=0.0, max_tokens=500).with_structured_output(RouteDecision)
judge_llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY, temperature=0.0, max_tokens=500).with_structured_output(RagJudge)
answer_llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY, temperature=0.7, max_tokens=500)

class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    route: Literal["rag", "web", "answer", "end"]
    rag:str
    web:str
    web_search_enabled: bool


#Node: for individual functions
#NODE: router(decision node)
def router_node(state:AgentState)-> AgentState:
    """Router node to decide the next action based on the conversation context"""
    print("Router node invoked")
    query=next( m.content for m in reversed(state["messages"]) if isinstance(m,HumanMessage))
    if isinstance(m,HumanMessage):
        for m in reversed(state["messages"]):
            next(m.content)

    else:        
    messages = state.get("messages", [])
    response = router_llm(messages)
    decision = response.output
    state["route"] = decision.route
    if decision.route == "end":
        state["rag"] = None
        state["web"] = None
    return state 
