import os
from openai import OpenAI

def quick_test():
    """Quick API key test"""
    print("🔑 Quick OpenAI API Key Test")
    print("=" * 40)
    
    # Get API key from user input
    api_key = input("Enter your NEW OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Simple test
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'test successful'"}],
            max_tokens=10
        )
        
        print("✅ API KEY WORKS!")
        print(f"Response: {response.choices[0].message.content}")
        
        # Save to environment file
        with open('.env', 'w') as f:
            f.write(f"OPENAI_API_KEY={api_key}")
        print("✅ API key saved to .env file")
        
    except Exception as e:
        print(f"❌ Still not working: {e}")
        if "401" in str(e):
            print("💡 Key is still invalid - check your OpenAI dashboard")
        elif "quota" in str(e).lower():
            print("💡 Quota exceeded - add billing to your OpenAI account")

if __name__ == "__main__":
    quick_test()
