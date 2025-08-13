import os
import openai
from openai import OpenAI

def test_api_key():
    """Test OpenAI API key functionality"""
    print("=" * 50)
    print("OpenAI API Key Diagnostic Test")
    print("=" * 50)
    
    # Check if API key is set in environment
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ API key found in environment (length: {len(api_key)})")
        print(f"✅ Key starts with: {api_key[:10]}...")
        print(f"✅ Key ends with: ...{api_key[-10:]}")
    else:
        print("❌ API key NOT found in environment")
        return False
    
    # Test direct API key usage
    try:
        print("\n🔄 Testing OpenAI connection...")
        client = OpenAI(api_key=api_key)
        
        # Test with a simple models list call
        print("📋 Fetching available models...")
        models = client.models.list()
        print(f"✅ Connection successful! Found {len(models.data)} models")
        
        # Show first few models
        print("📝 Available models (first 5):")
        for i, model in enumerate(models.data[:5]):
            print(f"   {i+1}. {model.id}")
        
        # Test a simple completion
        print("\n🤖 Testing chat completion...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'API test successful' if you can read this."}
            ],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        print(f"✅ Chat completion test: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        
        # Check for common error types
        error_str = str(e).lower()
        if "invalid api key" in error_str or "unauthorized" in error_str:
            print("💡 This looks like an API key authentication issue")
            print("   - Check if the API key is correct")
            print("   - Verify the key hasn't expired")
            print("   - Make sure you have sufficient credits")
        elif "rate limit" in error_str:
            print("💡 Rate limit exceeded - try again in a moment")
        elif "quota" in error_str:
            print("💡 Quota exceeded - check your OpenAI billing")
        elif "network" in error_str or "connection" in error_str:
            print("💡 Network connection issue")
        
        return False

if __name__ == "__main__":
    # Set API key for this test
    test_key = "your_openai_api_key_here"  # Replace with your actual API key
    os.environ["OPENAI_API_KEY"] = test_key
    
    success = test_api_key()
    
    if success:
        print("\n🎉 Your API key is working correctly!")
        print("🚀 Ready to start the data analyst agent")
    else:
        print("\n❌ API key test failed")
        print("📞 Please check your OpenAI account and API key")
