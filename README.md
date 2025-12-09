# Issue Grouping

export UV_CACHE_DIR='/path/to/your/cache'

API starting Command:

```python 
 gunicorn -w 4 -k uvicorn.workers.UvicornWorker --max-requests 1 --max-requests-jitter 0  -b 0.0.0.0:8001 "api:app" --graceful-timeout 30 --keep-alive 5
```