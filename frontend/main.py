from fastapi import FastAPI
from reactpy.backend.fastapi import configure
from reactpy import component, html
import uvicorn

app = FastAPI()


@component
def HelloWorld():
    return html.h1("B.R.U.T.U.S")


configure(app, HelloWorld)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run with:
# uvicorn main:app --reload
