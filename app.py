import os
import json
import urllib.parse
from typing import List, Dict, Any
from dotenv import load_dotenv
from litellm import completion
import yfinance as yf
import streamlit as st
from openai import OpenAI


load_dotenv()
client = OpenAI()

st.markdown("""
    <style>
    div[data-testid="stForm"] {
        border: none;
        padding: 0px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. ENVIRONMENT (E)
# ==========================================
class Environment:
    def get_stock_data(self, ticker: str) -> str:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            info = stock.info
            
            logo_url = ""
            website = info.get('website', '')
            if website:
                parsed_url = urllib.parse.urlparse(website)
                domain = parsed_url.netloc.replace('www.', '')
                if domain:
                    logo_url = f"https://cdn.tickerlogos.com/{domain}"
            
            if logo_url:
                st.session_state["logo_url"] = logo_url
            
            stock_data = {
                "company_name": info.get('longName', ticker),
                "current_price": info.get('currentPrice', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "market_cap": info.get('marketCap', 'N/A'),
                "pe_ratio": info.get('trailingPE', 'N/A'),
                "summary": info.get('longBusinessSummary', 'N/A'),
                "recent_history": hist.tail().to_string() if not hist.empty else "No history available"
            }
            return json.dumps(stock_data)
        except Exception as e:
            return f"Error fetching stock data for {ticker}: {str(e)}"

# ==========================================
# 2. ACTIONS (A)
# ==========================================
class ActionRegistry:
    def __init__(self, env: Environment):
        self.env = env
        self.functions = {
            "get_stock_data": self.env.get_stock_data,
            "terminate": self.terminate
        }

    def terminate(self, message: str) -> str:
        return f"TERMINATE:{message}"

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_stock_data",
                    "description": "Fetches live stock data, financials, and historical prices for a given ticker symbol.",
                    "parameters": {
                        "type": "object",
                        "properties": {"ticker": {"type": "string", "description": "The stock ticker symbol, e.g. AAPL, TSLA"}},
                        "required": ["ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "terminate",
                    "description": "Terminates the conversation and provides the final financial analysis summary.",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"]
                    }
                }
            }
        ]

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Any:
        if tool_name in self.functions:
            try:
                return self.functions[tool_name](**args)
            except Exception as e:
                return f"Error executing {tool_name}: {str(e)}"
        return f"Unknown tool: {tool_name}"

# ==========================================
# 3. GOALS (G)
# ==========================================
class Goals:
    @staticmethod
    def get_system_prompt() -> str:
        return """
You are an expert AI stock analysis agent. 
When the user asks about a company or stock, use the "get_stock_data" tool to gather market data and metrics.
Analyze the financial data thoroughly, evaluate its performance, trends, and risk factors, and present a clear breakdown to the user.
When you are completely done, use the "terminate" tool to deliver your final summary.
"""

# ==========================================
# 4. MEMORY (M)
# ==========================================
class Memory:
    def __init__(self, initial_task: str):
        self.messages: List[Dict[str, Any]] = [
            {"role": "user", "content": initial_task}
        ]

    def add_assistant_action(self, action_dict: dict):
        self.messages.append({"role": "assistant", "content": json.dumps(action_dict)})

    def add_tool_result(self, result: Any):
        self.messages.append({"role": "user", "content": json.dumps({"result": result})})

    def get_full_context(self, system_prompt: str) -> List[Dict[str, Any]]:
        return [{"role": "system", "content": system_prompt}] + self.messages

# ==========================================
# GAME Agent Orchestrator Loop
# ==========================================
class GameAgent:
    def __init__(self, task: str):
        self.env = Environment()
        self.actions = ActionRegistry(self.env)
        self.memory = Memory(task)
        self.max_iterations = 10

    def run(self):
        iterations = 0
        final_output = "No analysis generated."
        while iterations < self.max_iterations:
            iterations += 1
            messages = self.memory.get_full_context(Goals.get_system_prompt())

            response = completion(
                model="openai/gpt-4o",
                messages=messages,
                tools=self.actions.get_tool_definitions(),
                max_tokens=1024
            )

            message = response.choices[0].message

            if message.tool_calls:
                tool = message.tool_calls[0]
                tool_name = tool.function.name
                tool_args = json.loads(tool.function.arguments)
                
                result = self.actions.execute(tool_name, tool_args)

                if tool_name == "terminate":
                    final_output = tool_args.get('message', 'Done')
                    break

                self.memory.add_assistant_action({"tool_name": tool_name, "args": tool_args})
                self.memory.add_tool_result(result)
            else:
                final_output = message.content
                break
        return final_output

# ==========================================
# THE UI
st.title("📈 AI Stock Analysis Agent")
st.write("Run your GAME agent via a graphical web interface.")

# Form allows pressing 'Enter' to submit, CSS above hides the border box
with st.form(key="stock_form"):
    ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):")
    submit_button = st.form_submit_button(label="Run Analysis")

if submit_button:
    if not ticker_input:
        st.warning("Please enter a valid stock ticker.")
    else:
        if "logo_url" in st.session_state:
            del st.session_state["logo_url"]
            
        with st.spinner(f"Agent is analyzing {ticker_input.upper()}..."):
            agent = GameAgent(f"Analyze the stock {ticker_input}")
            result = agent.run()
            st.session_state["last_result"] = result

if "last_result" in st.session_state:
    st.markdown("---")
    
    if "logo_url" in st.session_state and st.session_state["logo_url"]:
        st.image(st.session_state["logo_url"], width=80)
        
    st.subheader("Analysis Results")
    st.write(st.session_state["last_result"])
    
    tts_col1, tts_col2 = st.columns([2, 1])
    
    with tts_col1:
        voice_style = st.selectbox(
            "Voice Style", 
            ["Man", "Woman"],
            label_visibility="collapsed"
        )
    
    with tts_col2:
        if st.button("📢 Read Aloud"):
            with st.spinner("Generating voice..."):
                selected_voice = "onyx" if "Man" in voice_style else "nova"
                speech_response = client.audio.speech.create(
                    model="tts-1",
                    voice=selected_voice,
                    input=st.session_state["last_result"]
                )
                st.audio(speech_response.content, format="audio/mp3")