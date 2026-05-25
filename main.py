import os

import uvicorn

PORT = os.getenv("PORT", 8000)

if __name__ == "__main__":
    uvicorn.run("backend.main:app", reload=True, host="0.0.0.0", port=int(PORT))