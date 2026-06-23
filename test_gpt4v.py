# test_gpt4v.py
import os
from providers import make_client
from dotenv import load_dotenv

# Load your environment variables (same as your main app)
load_dotenv()
client = make_client()

def test_gpt4v_access():
    print("🔍 Testing GPT-4V access...")
    print(f"🔑 Using API key: {os.getenv('OPENAI_API_KEY')[:20]}...")
    
    try:
        # This is a tiny 1x1 pixel red image encoded in base64
        # We use this so we don't need to upload a real image file
        test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        print("📤 Sending test request to OpenAI...")
        
        response = client.chat.completions.create(
            model="gpt-4o",  # This model supports vision
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{test_image}"}
                        }
                    ]
                }
            ],
            max_tokens=50
        )
        
        print("✅ GPT-4V Access: SUCCESS!")
        print(f"📝 AI Response: {response.choices[0].message.content}")
        print("🎉 Your API key has GPT-4V access!")
        return True
        
    except Exception as e:
        error_message = str(e)
        print("❌ GPT-4V Access: FAILED!")
        print(f"🚨 Error Details: {error_message}")
        
        # Give specific help based on the error
        if "does not exist" in error_message:
            print("\n💡 SOLUTION: Your account doesn't have access to GPT-4V models")
            print("   → Visit: https://platform.openai.com/account/billing")
            print("   → Upgrade to a paid plan")
            print("   → Request GPT-4 API access")
            
        elif "insufficient_quota" in error_message:
            print("\n💡 SOLUTION: You've run out of API credits")
            print("   → Visit: https://platform.openai.com/account/billing")
            print("   → Add credits to your account")
            
        elif "rate_limit" in error_message:
            print("\n💡 SOLUTION: You're making requests too quickly")
            print("   → Wait a few minutes and try again")
            
        elif "invalid_api_key" in error_message:
            print("\n💡 SOLUTION: Your API key is invalid")
            print("   → Check your .env file")
            print("   → Regenerate your API key on OpenAI dashboard")
        else:
            print(f"\n❓ Unknown error. Full details: {error_message}")
        
        return False

if __name__ == "__main__":
    print("🧪 GPT-4V Access Test Starting...")
    print("=" * 50)
    
    # Check if API key exists first
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OPENAI_API_KEY found in environment!")
        print("💡 Make sure your .env file contains: OPENAI_API_KEY=your_key_here")
    else:
        test_gpt4v_access()
    
    print("=" * 50)
    print("🏁 Test Complete!")