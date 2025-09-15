from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI(
    title="IssueGrouping",
    contact={"name": "Mohammed Altaf", "email": "altaf@qti.qualcomm.com"},
    docs_url="/api",
    redoc_url=None,
)


@app.get("/docs", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/api")


@app.get("/api/get_error_cluster_name")
async def get_error_cluster_name():
    """
    This API provides the cluster name to which the error belongs to.
    """
    return {"errorClusterName": "Issue Grouping is not yet implemented."}
