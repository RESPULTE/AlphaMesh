import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

load_dotenv()

auth = JWTVerifier(
    jwks_uri=f"{os.getenv('STYTCH_DOMAIN')}/.well-known/jwks.json",
    issuer=os.getenv("STYTCH_DOMAIN"),
    algorithm="RS256",
    audience=os.getenv("STYTCH_PROJECT_ID"),
)
app = FastMCP(name="testing", auth=auth)


@app.tool()
def get_notes(_ctx):
    return "no notes"


@app.tool()
def add_note(_ctx, content):
    return f"added note: {content}"


@app.custom_route("/.well-known/oauth-protected-resource", methods=["GET", "OPTIONS"])
def oauth_metadata(request: Request):
    base_url = str(request.base_url).rstrip("/")
    return JSONResponse(
        {
            "resource": f"{base_url}/mcp",
            "authorization_servers": [os.getenv("STYTCH_DOMAIN")],
            "scopes_supported": ["read", "write"],
            "bearer_methods_supported": ["header", "body"],
        }
    )


if __name__ == "__main__":
    app.run(
        transport="http",
        host="127.0.0.1",
        port=8000,
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_credentials=True,
                allow_headers=["*"],
            )
        ],
    )
