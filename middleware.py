from flask import request

def prefix_middleware(app):
    @app.before_request
    def rewrite_prefix():
        if request.path.startswith("/app"):
            request.environ["PATH_INFO"] = request.path.replace("/app", "", 1)