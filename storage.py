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
    Upload bytes to Supabase Storage with better duplicate handling.
    """
    try:
        print(f"🔄 Attempting to upload to: {path}")
        
        # Try to upload the file
        result = supabase.storage.from_(bucket).upload(path, file_bytes)
        print(f"✅ Upload successful: {result}")
        
    except StorageApiError as e:
        print(f"📝 Storage error details: {e}")
        
        # If it's a duplicate (409), try to update instead
        if getattr(e, "statusCode", None) == 409 or "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            print(f"🔄 File exists, updating instead...")
            try:
                # Update the existing file
                result = supabase.storage.from_(bucket).update(path, file_bytes)
                print(f"✅ Update successful: {result}")
            except Exception as update_error:
                print(f"❌ Update failed: {update_error}")
                # If update fails, try remove then upload
                try:
                    print(f"🗑️ Removing existing file and re-uploading...")
                    supabase.storage.from_(bucket).remove([path])
                    result = supabase.storage.from_(bucket).upload(path, file_bytes)
                    print(f"✅ Remove and re-upload successful: {result}")
                except Exception as final_error:
                    print(f"💥 All upload methods failed: {final_error}")
                    raise Exception(f"Failed to upload {path}: {final_error}")
        else:
            # If it's not a duplicate error, re-raise
            print(f"💥 Non-duplicate storage error: {e}")
            raise Exception(f"Storage error for {path}: {e}")
    
    except Exception as e:
        print(f"💥 Unexpected upload error: {e}")
        raise Exception(f"Upload error for {path}: {e}")
    
    # Return the public URL
    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"
    print(f"🌐 Public URL: {public_url}")
    return public_url