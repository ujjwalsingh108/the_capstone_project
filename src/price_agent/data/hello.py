import modal
from modal import Image

image = Image.debian_slim().pip_install(
    "requests", 
    "uvicorn", 
    "fastapi", 
    "pydantic_settings",
    "datasets"
    )
app = modal.App("hello")

@app.function(image=image)
def hello() -> str:
    import requests
    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    return f"Hello from {data['city']}, {data['region']}, {data['country']}!!"

@app.function(image=image, region="eu")
def hello_europe() -> str:
    import requests
    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    return f"Hello from {data['city']}, {data['region']}, {data['country']}!!"

# Only run if executed directly, not when imported
if __name__ == "__main__":
    with app.run():
        print("US / Default:", hello.remote())
        print("Europe:", hello_europe.remote())