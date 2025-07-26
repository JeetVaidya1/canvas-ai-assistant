# storage.py

import os
from supabase import create_client
from dotenv import load_dotenv
from storage3.exceptions import StorageApiError

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_file(bucket: str, file_bytes: bytes, path: str) -> str:
    """
    Upload bytes to Supabase Storage. 
    If the file already exists (409), ignore and proceed.
    """
    try:
        supabase.storage.from_(bucket).upload(path, file_bytes)
    except StorageApiError as e:
        # If it’s a duplicate, swallow it; otherwise re-raise
        if getattr(e, "statusCode", None) != 409:
            raise
    # Return the public URL (bucket is public in the UI)
    return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"
