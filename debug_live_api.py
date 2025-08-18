# debug_live_api.py - Test the live running server
import requests
import json
import sys

def test_server_endpoints():
    """Test each endpoint with actual requests"""
    print("🧪 Testing live server endpoints...")
    
    base_url = "http://localhost:8000"
    
    # Test basic server health
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ Server health check: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        return False
    
    # Test exam status endpoint
    try:
        response = requests.get(f"{base_url}/api/exam-status", timeout=5)
        print(f"📊 Exam status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        elif response.status_code == 404:
            print("   ❌ Exam status endpoint not found - add it to main.py")
        else:
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Exam status test failed: {e}")
    
    # Test past papers endpoint
    try:
        response = requests.get(f"{base_url}/api/past-papers/test_course", timeout=5)
        print(f"📄 Past papers: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        elif response.status_code == 404:
            print("   ❌ Past papers endpoint not found")
        else:
            print(f"   Error: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Past papers test failed: {e}")
    
    # Test exam generation endpoint (with minimal data)
    try:
        form_data = {
            'course_id': 'test_course',
            'exam_type': 'practice',
            'question_count': '3',
            'time_limit': '60',
            'difficulty': 'easy',
            'question_types': '["multiple_choice"]',
            'user_id': 'test_user'
        }
        
        response = requests.post(f"{base_url}/api/generate-practice-exam", data=form_data, timeout=10)
        print(f"🎯 Exam generation: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success':
                print(f"   ✅ Generated exam with {len(result['exam']['questions'])} questions")
            else:
                print(f"   ⚠️ Generation failed: {result.get('message')}")
        elif response.status_code == 404:
            print("   ❌ Generate exam endpoint not found")
        elif response.status_code == 422:
            print("   ⚠️ Validation error (missing required fields)")
            print(f"   Details: {response.text[:300]}")
        else:
            print(f"   Error {response.status_code}: {response.text[:300]}")
            
    except Exception as e:
        print(f"❌ Exam generation test failed: {e}")
    
    # Test upload endpoint (without actual file)
    try:
        response = requests.post(f"{base_url}/api/upload-past-paper", timeout=5)
        print(f"📤 Upload endpoint: {response.status_code}")
        if response.status_code == 422:
            print("   ✅ Endpoint exists (422 = missing required file)")
        elif response.status_code == 404:
            print("   ❌ Upload endpoint not found")
        else:
            print(f"   Unexpected: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
    
    return True

def check_frontend_requests():
    """Check if frontend can reach the server"""
    print("\n🌐 Frontend connectivity check...")
    
    # Test CORS
    headers = {
        'Origin': 'http://localhost:3000',  # Typical React dev server
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'Content-Type'
    }
    
    try:
        response = requests.options("http://localhost:8000/api/generate-practice-exam", headers=headers, timeout=5)
        print(f"✅ CORS preflight: {response.status_code}")
        
        # Check CORS headers
        cors_headers = {k: v for k, v in response.headers.items() if 'access-control' in k.lower()}
        if cors_headers:
            print("   CORS headers:", cors_headers)
        else:
            print("   ⚠️ No CORS headers found")
            
    except Exception as e:
        print(f"❌ CORS check failed: {e}")

def test_course_setup():
    """Test if we have any courses set up"""
    print("\n🏫 Course setup check...")
    
    try:
        response = requests.get("http://localhost:8000/list-courses", timeout=5)
        print(f"📚 List courses: {response.status_code}")
        
        if response.status_code == 200:
            courses = response.json().get('courses', [])
            print(f"   Found {len(courses)} courses")
            
            if courses:
                # Test first course for files
                course_id = courses[0]['course_id']
                file_response = requests.get(f"http://localhost:8000/list-files?course_id={course_id}", timeout=5)
                if file_response.status_code == 200:
                    files = file_response.json().get('files', [])
                    print(f"   Course '{course_id}' has {len(files)} files")
                    if len(files) == 0:
                        print("   ⚠️ No files uploaded - this might cause exam generation to fail")
                else:
                    print(f"   Couldn't check files: {file_response.status_code}")
            else:
                print("   ⚠️ No courses found - create a course first")
        else:
            print(f"   Error: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Course check failed: {e}")

def simulate_frontend_request():
    """Simulate the exact request the frontend would make"""
    print("\n🎭 Simulating frontend request...")
    
    # This simulates what your React component does
    form_data = {
        'course_id': 'test_course',  # You might need to use a real course ID
        'exam_type': 'practice',
        'question_count': '10',
        'time_limit': '120',
        'difficulty': 'mixed',
        'question_types': '["multiple_choice", "calculation", "short_answer"]',
        'topic_focus': '',
        'user_id': 'anonymous'
    }
    
    try:
        print(f"Sending request to: /api/generate-practice-exam")
        print(f"Form data: {form_data}")
        
        response = requests.post(
            "http://localhost:8000/api/generate-practice-exam", 
            data=form_data,
            timeout=30  # Give it more time
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('message', 'No message')}")
            if 'exam' in result:
                print(f"   Generated {len(result['exam']['questions'])} questions")
        else:
            print(f"❌ Failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Simulation failed: {e}")

def main():
    print("🔍 Detailed API Debug Tool")
    print("=" * 50)
    
    if not test_server_endpoints():
        print("\n❌ Server is not running! Start with: uvicorn main:app --reload")
        sys.exit(1)
    
    check_frontend_requests()
    test_course_setup()
    simulate_frontend_request()
    
    print("\n" + "=" * 50)
    print("🎯 DEBUG COMPLETE")
    print("\nIf you're still getting 404s in the frontend:")
    print("1. Check your React app is making requests to the right URL")
    print("2. Make sure CORS is working (check browser dev tools)")
    print("3. Verify the course_id exists and has files")
    print("4. Check browser network tab for exact error details")

if __name__ == "__main__":
    main()