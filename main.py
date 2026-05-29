from fastapi import FastAPI

app = FastAPI(name="rag-chatbot")

@app.get("/health")
async def health_check():
    return {"message": "Hello from rag-chatbot!"}



def main():
    print("Hello from rag-chatbot!")


if __name__ == "__main__":
    main()
