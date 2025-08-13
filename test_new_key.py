import os
from openai import OpenAI

def test_new_key():
    """Test the new API key"""
    print("🔑 Testing New OpenAI API Key")
    print("=" * 50)
    
    # Set the new API key
    new_key = "your_openai_api_key_here"  # Replace with your actual API key
    os.environ["OPENAI_API_KEY"] = new_key
    
    print(f"✅ API key set (length: {len(new_key)})")
    print(f"✅ Key format looks correct (starts with sk-proj-)")
    
    try:
        client = OpenAI(api_key=new_key)
        
        print("\n🔄 Testing connection...")
        
        # Test with models list
        models = client.models.list()
        print(f"✅ Connection successful! Found {len(models.data)} models")
        
        # Test chat completion
        print("\n🤖 Testing chat completion...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'API test successful' if you can read this."}
            ],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        print(f"✅ Chat completion: {result}")
        
        # Save to .env file
        with open('.env', 'w') as f:
            f.write(f"OPENAI_API_KEY={new_key}")
        print("✅ API key saved to .env file")
        
        print("\n🎉 SUCCESS! Your new API key is working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        error_str = str(e).lower()
        if "401" in error_str or "unauthorized" in error_str:
            print("💡 API key still invalid - check OpenAI dashboard")
        elif "quota" in error_str or "billing" in error_str:
            print("💡 Billing/quota issue - add payment method to OpenAI account")
        elif "rate limit" in error_str:
            print("💡 Rate limit - wait a moment and try again")
        
        return False

if __name__ == "__main__":
    success = test_new_key()
    
    if success:
        print("\n🚀 Ready to start the Data Analyst Agent!")
    else:
        print("\n❌ API key test failed - check your OpenAI account")
