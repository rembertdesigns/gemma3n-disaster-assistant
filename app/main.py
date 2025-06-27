from app.inference import run_disaster_analysis
from app.preprocessing import preprocess_input
from app.utils import load_resources

def main():
    print("🔧 Initializing Disaster Assistant...\n")

    user_input = input("Enter your report (text/voice/image path): ")
    prepped = preprocess_input(user_input)
    
    results = run_disaster_analysis(prepped)
    print("\n✅ Analysis Result:")
    print(results)

if __name__ == "__main__":
    main()
